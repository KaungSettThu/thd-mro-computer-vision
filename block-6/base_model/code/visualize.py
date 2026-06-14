import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from torch.utils.data import DataLoader

from dataset import GalleryDataset, EVAL_TRANSFORMS
from model import get_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLORS = plt.cm.tab10.colors


def build_all_identities(train_root, gallery_root, query_root):
    """
    Combine all images from all identities into one pool.
    Train identities (1-5) labeled as 'seen_X', test identities (6-7) as 'unseen_X'.
    """
    train_ds = GalleryDataset(train_root)
    gallery_ds = GalleryDataset(gallery_root)
    query_ds = GalleryDataset(query_root)

    samples = []
    label_map = {}

    for path, lbl in train_ds.samples:
        name = {v: k for k, v in train_ds.label_map.items()}[lbl]
        key = f'seen_{name}'
        label_map.setdefault(key, len(label_map))
        samples.append((path, label_map[key]))

    for path, lbl in gallery_ds.samples:
        name = {v: k for k, v in gallery_ds.label_map.items()}[lbl]
        key = f'unseen_{name}'
        label_map.setdefault(key, len(label_map))
        samples.append((path, label_map[key]))

    for path, lbl in query_ds.samples:
        name = {v: k for k, v in query_ds.label_map.items()}[lbl]
        key = f'unseen_{name}'
        label_map.setdefault(key, len(label_map))
        samples.append((path, label_map[key]))

    return samples, label_map


def extract_from_samples(model, samples):
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for path, label in samples:
            img = EVAL_TRANSFORMS(Image.open(path).convert('RGB')).unsqueeze(0).to(DEVICE)
            emb = model(img).cpu().numpy()[0]
            embeddings.append(emb)
            labels.append(label)
    return np.array(embeddings), np.array(labels)


def tsne_plot(ax, embeddings, labels, label_map, title):
    n = len(embeddings)
    perplexity = min(30, max(5, n // 4))
    proj = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000).fit_transform(embeddings)

    int_to_name = {v: k for k, v in label_map.items()}
    unique_labels = sorted(set(labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = int_to_name[lbl]
        is_unseen = name.startswith('unseen_')
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   color=COLORS[i % len(COLORS)],
                   label=name.replace('seen_', 'ID ').replace('unseen_', 'ID ') +
                         (' (unseen)' if is_unseen else ' (train)'),
                   marker='*' if is_unseen else 'o',
                   s=120 if is_unseen else 60,
                   alpha=0.9, edgecolors='k', linewidths=0.4)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.axis('off')


def run_visualization(train_root, gallery_root, query_root,
                      triplet_ckpt, contrastive_ckpt, embedding_dim, out_path):

    samples, label_map = build_all_identities(train_root, gallery_root, query_root)
    print(f"Total images for t-SNE: {len(samples)} across {len(label_map)} identities")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('t-SNE Embedding Space — All Identities\n'
                 'Circles = seen during training | Stars = unseen (test) identities',
                 fontsize=13, fontweight='bold')

    print("Plot 1: Untrained (random weights)...")
    model = get_model(embedding_dim=embedding_dim, pretrained=False).to(DEVICE)
    embs, lbls = extract_from_samples(model, samples)
    tsne_plot(axes[0], embs, lbls, label_map, 'Untrained\n(Random Weights)')

    print("Plot 2: Triplet loss model...")
    model = get_model(embedding_dim=embedding_dim, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(triplet_ckpt, map_location=DEVICE))
    embs, lbls = extract_from_samples(model, samples)
    tsne_plot(axes[1], embs, lbls, label_map, 'Trained: Triplet Loss')

    print("Plot 3: Contrastive loss model...")
    model = get_model(embedding_dim=embedding_dim, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(contrastive_ckpt, map_location=DEVICE))
    embs, lbls = extract_from_samples(model, samples)
    tsne_plot(axes[2], embs, lbls, label_map, 'Trained: Contrastive Loss')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='dataset/train')
    parser.add_argument('--gallery_root', default='dataset/gallery')
    parser.add_argument('--query_root', default='dataset/test')
    parser.add_argument('--triplet_ckpt', default='checkpoints/model_triplet.pth')
    parser.add_argument('--contrastive_ckpt', default='checkpoints/model_contrastive.pth')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--out', default='tsne_comparison.png')
    args = parser.parse_args()

    for ckpt in (args.triplet_ckpt, args.contrastive_ckpt):
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt} — run train.py first.")
            exit(1)

    run_visualization(args.train_root, args.gallery_root, args.query_root,
                      args.triplet_ckpt, args.contrastive_ckpt,
                      args.embedding_dim, args.out)
