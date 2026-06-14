import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_all(triplet_losses, contrastive_losses, hard_losses,
             base_metrics, model1_metrics, out_path):

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    epochs_base  = range(1, len(triplet_losses) + 1)
    epochs_hard  = range(1, len(hard_losses) + 1)

    # --- Plot 1: Base model loss curves ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_base, triplet_losses,     color='steelblue',  linewidth=2, label='Triplet (ResNet18)')
    ax1.plot(epochs_base, contrastive_losses, color='darkorange', linewidth=2, label='Contrastive (ResNet18)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Base Model — Training Loss', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xlim(1, len(triplet_losses))

    # --- Plot 2: Model 1 loss curve ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_hard, hard_losses, color='seagreen', linewidth=2, label='Hard Triplet (ResNet50)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Model 1 — Training Loss\n(Batch Hard Mining + Cosine Annealing)', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_xlim(1, len(hard_losses))

    # --- Plot 3: All loss curves overlaid (normalized to epoch range) ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs_base, triplet_losses,     color='steelblue',  linewidth=2, label='Base Triplet')
    ax3.plot(epochs_base, contrastive_losses, color='darkorange', linewidth=2, label='Base Contrastive')
    ax3.plot(epochs_hard, hard_losses,        color='seagreen',   linewidth=2, label='Model 1 Hard Triplet')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss')
    ax3.set_title('All Loss Curves Comparison', fontweight='bold')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # --- Plot 4: Rank-1 and mAP bar chart base vs model_1 ---
    ax4 = fig.add_subplot(gs[1, 0:2])
    models = ['Base\nTriplet', 'Base\nContrastive', 'Model 1\nHard Triplet']
    rank1_vals = [base_metrics['triplet']['rank1']     * 100,
                  base_metrics['contrastive']['rank1'] * 100,
                  model1_metrics['rank1']              * 100]
    map_vals   = [base_metrics['triplet']['map']       * 100,
                  base_metrics['contrastive']['map']   * 100,
                  model1_metrics['map']                * 100]

    x = np.arange(len(models))
    w = 0.35
    colors_r1  = ['steelblue', 'darkorange', 'seagreen']
    colors_map = ['#5B9BD5',   '#F4A261',    '#52B788']

    bars1 = ax4.bar(x - w/2, rank1_vals, w, label='Rank-1', color=colors_r1,  alpha=0.85)
    bars2 = ax4.bar(x + w/2, map_vals,   w, label='mAP',    color=colors_map, alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_xticks(x); ax4.set_xticklabels(models, fontsize=10)
    ax4.set_ylabel('Score (%)'); ax4.set_ylim(0, 120)
    ax4.set_title('Rank-1 & mAP — Base Model vs Model 1', fontweight='bold')
    ax4.legend(); ax4.grid(True, alpha=0.3, axis='y')

    # --- Plot 5: Distance metric comparison for model 1 ---
    ax5 = fig.add_subplot(gs[1, 2])
    dist_names = ['Cosine', 'Euclidean', 'Manhattan']
    r1_dist  = [model1_metrics['cosine_r1']    * 100,
                model1_metrics['euclidean_r1'] * 100,
                model1_metrics['manhattan_r1'] * 100]
    map_dist = [model1_metrics['cosine_map']    * 100,
                model1_metrics['euclidean_map'] * 100,
                model1_metrics['manhattan_map'] * 100]

    x2 = np.arange(len(dist_names))
    b1 = ax5.bar(x2 - w/2, r1_dist,  w, label='Rank-1', color=['#264653','#2A9D8F','#E9C46A'], alpha=0.85)
    b2 = ax5.bar(x2 + w/2, map_dist, w, label='mAP',    color=['#457B9D','#52B788','#F4A261'], alpha=0.85)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax5.set_xticks(x2); ax5.set_xticklabels(dist_names)
    ax5.set_ylabel('Score (%)'); ax5.set_ylim(0, 120)
    ax5.set_title('Distance Metric Comparison\n(Model 1 — L2-normalized embeddings)', fontweight='bold')
    ax5.legend(); ax5.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Re-Identification — Base Model vs Model 1 Summary', fontsize=15, fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--triplet_losses',     default='checkpoints/model_triplet_losses.json')
    parser.add_argument('--contrastive_losses', default='checkpoints/model_contrastive_losses.json')
    parser.add_argument('--hard_losses',        default='checkpoints/model1_hard_triplet_losses.json')
    parser.add_argument('--out', default='training_summary.png')
    args = parser.parse_args()

    for p in (args.triplet_losses, args.contrastive_losses, args.hard_losses):
        if not os.path.exists(p):
            print(f"Loss file not found: {p} — run train.py first.")
            exit(1)

    with open(args.triplet_losses)     as f: triplet_losses     = json.load(f)
    with open(args.contrastive_losses) as f: contrastive_losses = json.load(f)
    with open(args.hard_losses)        as f: hard_losses        = json.load(f)

    base_metrics = {
        'triplet':     {'rank1': 0.875, 'map': 0.778},
        'contrastive': {'rank1': 0.750, 'map': 0.709},
    }
    model1_metrics = {
        'rank1': 1.000, 'map': 0.846,
        'cosine_r1':    1.000, 'cosine_map':    0.846,
        'euclidean_r1': 1.000, 'euclidean_map': 0.846,
        'manhattan_r1': 0.875, 'manhattan_map': 0.832,
    }

    plot_all(triplet_losses, contrastive_losses, hard_losses,
             base_metrics, model1_metrics, args.out)
