import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_all(triplet_losses, contrastive_losses, metrics, out_path):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    epochs = range(1, len(triplet_losses) + 1)

    # --- Plot 1: Triplet loss curve ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, triplet_losses, color='steelblue', linewidth=2, label='Triplet Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss — Triplet Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(triplet_losses))

    # --- Plot 2: Contrastive loss curve ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, contrastive_losses, color='darkorange', linewidth=2, label='Contrastive Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss — Contrastive Loss', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(contrastive_losses))

    # --- Plot 3: Both loss curves overlaid ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, triplet_losses,     color='steelblue',  linewidth=2, label='Triplet')
    ax3.plot(epochs, contrastive_losses, color='darkorange', linewidth=2, label='Contrastive')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss Curves — Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, len(triplet_losses))

    # --- Plot 4: Metrics bar chart ---
    ax4 = fig.add_subplot(gs[1, 1])
    metric_names = ['Rank-1', 'Rank-5', 'mAP']
    triplet_vals    = [metrics['triplet']['rank1']    * 100,
                       metrics['triplet']['rank5']    * 100,
                       metrics['triplet']['map']      * 100]
    contrastive_vals = [metrics['contrastive']['rank1'] * 100,
                        metrics['contrastive']['rank5'] * 100,
                        metrics['contrastive']['map']   * 100]

    x = np.arange(len(metric_names))
    w = 0.32
    bars1 = ax4.bar(x - w/2, triplet_vals,     w, label='Triplet',     color='steelblue',  alpha=0.85)
    bars2 = ax4.bar(x + w/2, contrastive_vals, w, label='Contrastive', color='darkorange', alpha=0.85)

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                 f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names)
    ax4.set_ylabel('Score (%)')
    ax4.set_ylim(0, 115)
    ax4.set_title('Evaluation Metrics — Triplet vs Contrastive', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Base Model — Training & Evaluation Summary', fontsize=14, fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--triplet_losses',     default='checkpoints/model_triplet_losses.json')
    parser.add_argument('--contrastive_losses', default='checkpoints/model_contrastive_losses.json')
    parser.add_argument('--out', default='training_summary.png')
    args = parser.parse_args()

    for p in (args.triplet_losses, args.contrastive_losses):
        if not os.path.exists(p):
            print(f"Loss file not found: {p} — run train.py first.")
            exit(1)

    with open(args.triplet_losses)     as f: triplet_losses     = json.load(f)
    with open(args.contrastive_losses) as f: contrastive_losses = json.load(f)

    # hardcoded from evaluate.py output — update if you retrain
    metrics = {
        'triplet':     {'rank1': 0.75, 'rank5': 1.00, 'map': 0.771},
        'contrastive': {'rank1': 0.75, 'rank5': 1.00, 'map': 0.675},
    }

    plot_all(triplet_losses, contrastive_losses, metrics, args.out)
