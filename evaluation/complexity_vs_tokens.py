import json
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from scipy import stats

# ===== CONFIGURABLE TEXT SIZE SCALING =====
TEXT_SCALE = 1.5
# ==========================================

PATHS = {
    'grpo_original': 'grpo/original/msvamp_gemma3_12b_0.0_checkpoint.json',
    'grpo_augmented': 'grpo/augmented/dmsvamp_gemma3_12b_0.0_checkpoint.json',
    'qwen_original': 'baselines/original/msvamp_qwen3_8b_checkpoint.json',
    'qwen_augmented': 'baselines/augmented/msvamp_aug_qwen3_8b_checkpoint.json'
}

def extract_vop_from_graph(graph_output):
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', graph_output, re.DOTALL)
        if not json_match:
            return None
        graph_data = json.loads(json_match.group(1))
        nodes = graph_data.get('nodes', [])
        vop_count = sum(1 for node in nodes if node.get('op') not in ['const', 'identity'])
        return vop_count
    except:
        return None

def load_grpo_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        records = json.load(f)
    for record in records:
        if record.get('correct') == True and record.get('execution_success') == True:
            vop = extract_vop_from_graph(record.get('graph_output', ''))
            if vop is not None:
                data.append({
                    'index': record.get('index'),
                    'vop': vop,
                    'output_tokens': record.get('output_token')
                })
    return data

def load_qwen_data(filepath, grpo_data):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        records = json.load(f)
    grpo_map = {item['index']: item['vop'] for item in grpo_data}
    for record in records:
        if record.get('correct') == True:
            idx = record.get('id')
            if idx in grpo_map:
                data.append({
                    'index': idx,
                    'vop': grpo_map[idx],
                    'output_tokens': record.get('output_tokens')
                })
    return data

def compute_statistics_by_complexity(data):
    """Compute mean, std grouped by complexity level."""
    complexity_stats = {}
    for item in data:
        vop = item['vop']
        tokens = item['output_tokens']
        if vop not in complexity_stats:
            complexity_stats[vop] = []
        complexity_stats[vop].append(tokens)
    
    results = {}
    for vop, tokens_list in complexity_stats.items():
        results[vop] = {
            'mean': np.mean(tokens_list),
            'std': np.std(tokens_list),
            'min': np.min(tokens_list),
            'max': np.max(tokens_list),
            'count': len(tokens_list)
        }
    return results

def create_single_analysis_plot(grpo_data, qwen_data, title_suffix):
    """Create a single plot with left and right subplots."""
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Colors
    grpo_color = '#1E88E5'
    qwen_color = '#FB8C00'
    
    # Compute statistics
    grpo_stats = compute_statistics_by_complexity(grpo_data)
    qwen_stats = compute_statistics_by_complexity(qwen_data)
    
    # Extract arrays for regression
    grpo_vops = np.array([item['vop'] for item in grpo_data])
    grpo_tokens = np.array([item['output_tokens'] for item in grpo_data])
    qwen_vops = np.array([item['vop'] for item in qwen_data])
    qwen_tokens = np.array([item['output_tokens'] for item in qwen_data])
    
    # Compute regressions
    grpo_slope, grpo_intercept, grpo_r, _, _ = stats.linregress(grpo_vops, grpo_tokens)
    qwen_slope, qwen_intercept, qwen_r, _, _ = stats.linregress(qwen_vops, qwen_tokens)
    grpo_r2 = grpo_r ** 2
    qwen_r2 = qwen_r ** 2
    
    # ===== LEFT PLOT: Line plots with confidence bands =====
    grpo_vops_sorted = sorted(grpo_stats.keys())
    grpo_means = [grpo_stats[v]['mean'] for v in grpo_vops_sorted]
    grpo_stds = [grpo_stats[v]['std'] for v in grpo_vops_sorted]
    
    qwen_vops_sorted = sorted(qwen_stats.keys())
    qwen_means = [qwen_stats[v]['mean'] for v in qwen_vops_sorted]
    qwen_stds = [qwen_stats[v]['std'] for v in qwen_vops_sorted]
    
    # Plot GRPO
    ax_left.plot(grpo_vops_sorted, grpo_means, 'o-', color=grpo_color, 
                linewidth=2.5, markersize=10, label='Ours (GRPO)', zorder=3)
    ax_left.fill_between(grpo_vops_sorted,
                        [m - s for m, s in zip(grpo_means, grpo_stds)],
                        [m + s for m, s in zip(grpo_means, grpo_stds)],
                        alpha=0.3, color=grpo_color)
    
    # Plot GRPO regression
    x_range = np.linspace(min(grpo_vops_sorted), max(grpo_vops_sorted), 100)
    grpo_fit = grpo_slope * x_range + grpo_intercept
    ax_left.plot(x_range, grpo_fit, '--', color=grpo_color, linewidth=2, 
                alpha=0.7, label=f'Linear fit (R²={grpo_r2:.3f})')
    
    # Plot Qwen
    ax_left.plot(qwen_vops_sorted, qwen_means, 's-', color=qwen_color,
                linewidth=2.5, markersize=10, label='Qwen 3 Reasoning', zorder=3)
    ax_left.fill_between(qwen_vops_sorted,
                        [m - s for m, s in zip(qwen_means, qwen_stds)],
                        [m + s for m, s in zip(qwen_means, qwen_stds)],
                        alpha=0.3, color=qwen_color)
    
    # Plot Qwen regression
    x_range_qwen = np.linspace(min(qwen_vops_sorted), max(qwen_vops_sorted), 100)
    qwen_fit = qwen_slope * x_range_qwen + qwen_intercept
    ax_left.plot(x_range_qwen, qwen_fit, '--', color=qwen_color, linewidth=2,
                alpha=0.7, label=f'Linear fit (R²={qwen_r2:.3f})')
    
    # Labels with 1.2x bigger text
    ax_left.set_xlabel('Complexity |Vop|', fontsize=13*TEXT_SCALE*1.2, fontweight='bold')
    ax_left.set_ylabel('Output Tokens', fontsize=13*TEXT_SCALE*1.2, fontweight='bold')
    ax_left.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_left.tick_params(axis='both', which='major', labelsize=10*TEXT_SCALE)
    
    # Legend at top in single row - moved higher
    ax_left.legend(fontsize=9*TEXT_SCALE, frameon=True, fancybox=True, 
                  shadow=True, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.12))
    
    # ===== RIGHT PLOT: Box plots =====
    complexity_levels = sorted(set(grpo_vops_sorted) | set(qwen_vops_sorted))
    
    grpo_box_data = []
    qwen_box_data = []
    for vop in complexity_levels:
        grpo_tokens_at_vop = [item['output_tokens'] for item in grpo_data if item['vop'] == vop]
        qwen_tokens_at_vop = [item['output_tokens'] for item in qwen_data if item['vop'] == vop]
        grpo_box_data.append(grpo_tokens_at_vop)
        qwen_box_data.append(qwen_tokens_at_vop)
    
    positions = np.arange(len(complexity_levels))
    width = 0.35
    
    bp1 = ax_right.boxplot(grpo_box_data, positions=positions - width/2, widths=width*0.8,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(facecolor=grpo_color, alpha=0.7),
                           medianprops=dict(color='darkblue', linewidth=2),
                           whiskerprops=dict(color=grpo_color),
                           capprops=dict(color=grpo_color))
    
    bp2 = ax_right.boxplot(qwen_box_data, positions=positions + width/2, widths=width*0.8,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(facecolor=qwen_color, alpha=0.7),
                           medianprops=dict(color='darkred', linewidth=2),
                           whiskerprops=dict(color=qwen_color),
                           capprops=dict(color=qwen_color))
    
    # Labels with 1.2x bigger text
    ax_right.set_xlabel('Complexity |Vop|', fontsize=13*TEXT_SCALE*1.2, fontweight='bold')
    ax_right.set_ylabel('Output Tokens', fontsize=13*TEXT_SCALE*1.2, fontweight='bold')
    ax_right.set_xticks(positions)
    ax_right.set_xticklabels(complexity_levels)
    ax_right.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax_right.tick_params(axis='both', which='major', labelsize=10*TEXT_SCALE)
    
    # Legend at top in single row - moved higher
    ax_right.legend([bp1["boxes"][0], bp2["boxes"][0]], 
                   ['Ours (GRPO)', 'Qwen 3 Reasoning'],
                   fontsize=9*TEXT_SCALE, frameon=True, fancybox=True, 
                   shadow=True, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.12))
    
    plt.tight_layout()
    
    return fig, grpo_r2, qwen_r2, grpo_slope, qwen_slope

def print_statistics(grpo_data, qwen_data, name):
    """Print statistics for a single dataset."""
    
    print(f"\n{name}:")
    print("-" * 80)
    
    # Extract arrays
    grpo_vops = np.array([item['vop'] for item in grpo_data])
    grpo_tokens = np.array([item['output_tokens'] for item in grpo_data])
    qwen_vops = np.array([item['vop'] for item in qwen_data])
    qwen_tokens = np.array([item['output_tokens'] for item in qwen_data])
    
    # Regressions
    grpo_slope, grpo_intercept, grpo_r, _, _ = stats.linregress(grpo_vops, grpo_tokens)
    qwen_slope, qwen_intercept, qwen_r, _, _ = stats.linregress(qwen_vops, qwen_tokens)
    grpo_r2 = grpo_r ** 2
    qwen_r2 = qwen_r ** 2
    
    print(f"\n  GRPO (Ours):")
    print(f"    - Sample size: {len(grpo_data)}")
    print(f"    - Complexity range: {grpo_vops.min():.0f} to {grpo_vops.max():.0f}")
    print(f"    - Token range: {grpo_tokens.min():.0f} to {grpo_tokens.max():.0f}")
    print(f"    - Scaling: {grpo_slope:.1f} tokens/|Vop|")
    print(f"    - R²: {grpo_r2:.3f} {'(STRONG correlation)' if grpo_r2 > 0.5 else '(WEAK correlation)'}")
    print(f"    - Mean tokens: {grpo_tokens.mean():.0f} ± {grpo_tokens.std():.0f}")
    
    print(f"\n  Qwen 3 Reasoning:")
    print(f"    - Sample size: {len(qwen_data)}")
    print(f"    - Complexity range: {qwen_vops.min():.0f} to {qwen_vops.max():.0f}")
    print(f"    - Token range: {qwen_tokens.min():.0f} to {qwen_tokens.max():.0f}")
    print(f"    - Scaling: {qwen_slope:.1f} tokens/|Vop|")
    print(f"    - R²: {qwen_r2:.3f} {'(STRONG correlation)' if qwen_r2 > 0.5 else '(WEAK/NO correlation)'}")
    print(f"    - Mean tokens: {qwen_tokens.mean():.0f} ± {qwen_tokens.std():.0f}")
    
    efficiency = qwen_slope / grpo_slope if grpo_slope > 0 else float('inf')
    print(f"\n  EFFICIENCY GAIN: {efficiency:.1f}× more efficient")
    
    # Compute CV (coefficient of variation) by complexity level
    grpo_stats = compute_statistics_by_complexity(grpo_data)
    qwen_stats = compute_statistics_by_complexity(qwen_data)
    
    print(f"\n  VARIANCE ANALYSIS (Coefficient of Variation = std/mean):")
    common_vops = sorted(set(grpo_stats.keys()) & set(qwen_stats.keys()))
    for vop in common_vops:
        grpo_cv = grpo_stats[vop]['std'] / grpo_stats[vop]['mean'] if grpo_stats[vop]['mean'] > 0 else 0
        qwen_cv = qwen_stats[vop]['std'] / qwen_stats[vop]['mean'] if qwen_stats[vop]['mean'] > 0 else 0
        print(f"    |Vop|={vop}: GRPO CV={grpo_cv:.2f}, Qwen CV={qwen_cv:.2f}")

def main():
    print("Loading data...")
    
    grpo_orig = load_grpo_data(PATHS['grpo_original'])
    grpo_aug = load_grpo_data(PATHS['grpo_augmented'])
    qwen_orig = load_qwen_data(PATHS['qwen_original'], grpo_orig)
    qwen_aug = load_qwen_data(PATHS['qwen_augmented'], grpo_aug)
    
    print(f"GRPO Original: {len(grpo_orig)} correct instances")
    print(f"GRPO Augmented: {len(grpo_aug)} correct instances")
    print(f"Qwen Original: {len(qwen_orig)} correct instances")
    print(f"Qwen Augmented: {len(qwen_aug)} correct instances")
    
    # Create and save ORIGINAL plot
    print("\nCreating ORIGINAL MSVAMP analysis...")
    fig_orig, _, _, _, _ = create_single_analysis_plot(grpo_orig, qwen_orig, "Original")
    
    output_orig_png = 'msvamp_original_analysis.png'
    fig_orig.savefig(output_orig_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Original plot saved: {output_orig_png}")
    
    output_orig_pdf = 'msvamp_original_analysis.pdf'
    fig_orig.savefig(output_orig_pdf, bbox_inches='tight', facecolor='white')
    print(f"Original PDF saved: {output_orig_pdf}")
    
    # Create and save AUGMENTED plot
    print("\nCreating AUGMENTED MSVAMP analysis...")
    fig_aug, _, _, _, _ = create_single_analysis_plot(grpo_aug, qwen_aug, "Augmented")
    
    output_aug_png = 'msvamp_augmented_analysis.png'
    fig_aug.savefig(output_aug_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Augmented plot saved: {output_aug_png}")
    
    output_aug_pdf = 'msvamp_augmented_analysis.pdf'
    fig_aug.savefig(output_aug_pdf, bbox_inches='tight', facecolor='white')
    print(f"Augmented PDF saved: {output_aug_pdf}")
    
    # Print statistics
    print("\n" + "="*80)
    print("SEPARATE ANALYSIS: Original vs Augmented MSVAMP")
    print("="*80)
    
    print_statistics(grpo_orig, qwen_orig, 'MSVAMP Original')
    print_statistics(grpo_aug, qwen_aug, 'MSVAMP + Distractors')
    
    print("\n" + "="*80)
    print("OUTPUT FILES:")
    print("  - msvamp_original_analysis.png")
    print("  - msvamp_original_analysis.pdf")
    print("  - msvamp_augmented_analysis.png")
    print("  - msvamp_augmented_analysis.pdf")
    print("="*80)

if __name__ == "__main__":
    main()