import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import GalleryDataset
from model import get_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_embeddings(model, root):
    dataset = GalleryDataset(root)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    embeddings, labels, paths = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls, pths in loader:
            embs = model(imgs.to(DEVICE)).cpu().numpy()
            embeddings.append(embs)
            labels.extend(lbls.numpy().tolist())
            paths.extend(pths)
    return np.vstack(embeddings), np.array(labels), paths


def compute_rank_k(query_embs, query_labels, gallery_embs, gallery_labels, k=5):
    """Cosine similarity ranking. Returns rank-1 and rank-k accuracy."""
    # query_embs: (Q, D), gallery_embs: (G, D)
    sim = query_embs @ gallery_embs.T  # (Q, G)
    ranked = np.argsort(-sim, axis=1)  # descending

    rank1, rankk = 0, 0
    for i, (row, qlabel) in enumerate(zip(ranked, query_labels)):
        gallery_ranked_labels = gallery_labels[row]
        if gallery_ranked_labels[0] == qlabel:
            rank1 += 1
        if qlabel in gallery_ranked_labels[:k]:
            rankk += 1

    n = len(query_labels)
    return rank1 / n, rankk / n


def compute_map(query_embs, query_labels, gallery_embs, gallery_labels):
    """Mean Average Precision."""
    sim = query_embs @ gallery_embs.T
    ranked = np.argsort(-sim, axis=1)

    aps = []
    for i, (row, qlabel) in enumerate(zip(ranked, query_labels)):
        gallery_ranked_labels = gallery_labels[row]
        hits = (gallery_ranked_labels == qlabel).astype(float)
        if hits.sum() == 0:
            continue
        cumulative = np.cumsum(hits)
        precision_at_k = cumulative / (np.arange(len(hits)) + 1)
        ap = (precision_at_k * hits).sum() / hits.sum()
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def evaluate(model_path, query_root, gallery_root, embedding_dim=128, label='Model'):
    model = get_model(embedding_dim=embedding_dim, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    query_embs, query_labels, _ = extract_embeddings(model, query_root)
    gallery_embs, gallery_labels, _ = extract_embeddings(model, gallery_root)

    # Align label spaces: query labels from test/, gallery labels from gallery/
    # Both use sorted identity folder names mapped to ints independently,
    # so we re-map by folder name to ensure they match.
    query_dataset = GalleryDataset(query_root)
    gallery_dataset = GalleryDataset(gallery_root)

    # Build name->int maps and remap to a shared space
    q_names = {v: k for k, v in query_dataset.label_map.items()}   # int -> name
    g_names = {v: k for k, v in gallery_dataset.label_map.items()} # int -> name

    shared_map = {}
    for name in set(q_names.values()) | set(g_names.values()):
        shared_map[name] = len(shared_map)

    query_labels_shared = np.array([shared_map[q_names[l]] for l in query_labels])
    gallery_labels_shared = np.array([shared_map[g_names[l]] for l in gallery_labels])

    rank1, rank5 = compute_rank_k(query_embs, query_labels_shared,
                                   gallery_embs, gallery_labels_shared, k=5)
    map_score = compute_map(query_embs, query_labels_shared,
                            gallery_embs, gallery_labels_shared)

    print(f"\n=== {label} ===")
    print(f"  Rank-1 accuracy : {rank1 * 100:.1f}%")
    print(f"  Rank-5 accuracy : {rank5 * 100:.1f}%")
    print(f"  mAP             : {map_score * 100:.1f}%")
    return rank1, rank5, map_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_root', default='dataset/test')
    parser.add_argument('--gallery_root', default='dataset/gallery')
    parser.add_argument('--embedding_dim', type=int, default=128)
    args = parser.parse_args()

    if os.path.exists('checkpoints/model_triplet.pth'):
        evaluate('checkpoints/model_triplet.pth',
                 args.query_root, args.gallery_root,
                 args.embedding_dim, label='Triplet Loss Model')

    if os.path.exists('checkpoints/model_contrastive.pth'):
        evaluate('checkpoints/model_contrastive.pth',
                 args.query_root, args.gallery_root,
                 args.embedding_dim, label='Contrastive Loss Model')
