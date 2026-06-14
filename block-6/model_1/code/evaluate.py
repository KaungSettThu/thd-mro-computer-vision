import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import GalleryDataset
from model import get_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_embeddings(model, root, normalize=True):
    dataset = GalleryDataset(root)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    embeddings, labels, paths = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls, pths in loader:
            embs = model(imgs.to(DEVICE), normalize=normalize).cpu().numpy()
            embeddings.append(embs)
            labels.extend(lbls.numpy().tolist())
            paths.extend(pths)
    return np.vstack(embeddings), np.array(labels), paths, dataset.label_map


def align_labels(q_lmap, g_lmap, q_labels, g_labels):
    q_int2name = {v: k for k, v in q_lmap.items()}
    g_int2name = {v: k for k, v in g_lmap.items()}
    shared = {}
    for name in set(q_int2name.values()) | set(g_int2name.values()):
        shared.setdefault(name, len(shared))
    q_s = np.array([shared[q_int2name[l]] for l in q_labels])
    g_s = np.array([shared[g_int2name[l]] for l in g_labels])
    return q_s, g_s


def rank_by_cosine(q_embs, g_embs):
    """Cosine similarity — higher is more similar. Works on L2-normalized embeddings."""
    sim = q_embs @ g_embs.T
    return np.argsort(-sim, axis=1)   # descending


def rank_by_euclidean(q_embs, g_embs):
    """Euclidean (L2) distance — lower is more similar."""
    diff = q_embs[:, None, :] - g_embs[None, :, :]   # (Q, G, D)
    dist = np.linalg.norm(diff, axis=2)               # (Q, G)
    return np.argsort(dist, axis=1)                   # ascending


def rank_by_manhattan(q_embs, g_embs):
    """Manhattan (L1) distance — lower is more similar."""
    diff = q_embs[:, None, :] - g_embs[None, :, :]
    dist = np.abs(diff).sum(axis=2)
    return np.argsort(dist, axis=1)


def compute_metrics(ranked, query_labels, gallery_labels, k=5):
    rank1, rankk, aps = 0, 0, []
    for i, (row, qlabel) in enumerate(zip(ranked, query_labels)):
        g_ranked = gallery_labels[row]
        if g_ranked[0] == qlabel:
            rank1 += 1
        if qlabel in g_ranked[:k]:
            rankk += 1
        hits = (g_ranked == qlabel).astype(float)
        if hits.sum() > 0:
            cumulative = np.cumsum(hits)
            prec_at_k = cumulative / (np.arange(len(hits)) + 1)
            aps.append((prec_at_k * hits).sum() / hits.sum())
    n = len(query_labels)
    return rank1 / n, rankk / n, float(np.mean(aps)) if aps else 0.0


def evaluate(model_path, query_root, gallery_root, embedding_dim=128,
             backbone='resnet18', label='Model', compare_metrics=False):

    model = get_model(embedding_dim=embedding_dim, pretrained=False, backbone=backbone).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Always evaluate with L2-normalized embeddings
    q_embs, q_labels, _, q_lmap = extract_embeddings(model, query_root, normalize=True)
    g_embs, g_labels, _, g_lmap = extract_embeddings(model, gallery_root, normalize=True)
    q_labels_s, g_labels_s = align_labels(q_lmap, g_lmap, q_labels, g_labels)

    print(f"\n=== {label} ===")

    if compare_metrics:
        # Compare cosine, euclidean, manhattan on L2-normalized embeddings
        metrics_table = {}
        for name, ranked in [
            ('Cosine similarity',   rank_by_cosine(q_embs, g_embs)),
            ('Euclidean distance',  rank_by_euclidean(q_embs, g_embs)),
            ('Manhattan distance',  rank_by_manhattan(q_embs, g_embs)),
        ]:
            r1, r5, map_s = compute_metrics(ranked, q_labels_s, g_labels_s)
            metrics_table[name] = (r1, r5, map_s)
            print(f"  [{name:22s}]  Rank-1={r1*100:.1f}%  Rank-5={r5*100:.1f}%  mAP={map_s*100:.1f}%")

        print(f"\n  Note: Cosine and Euclidean give identical rankings on L2-normalized embeddings.")
        print(f"  Manhattan may differ as it measures distance differently (sum of absolute differences).")
        return metrics_table
    else:
        ranked = rank_by_cosine(q_embs, g_embs)
        r1, r5, map_s = compute_metrics(ranked, q_labels_s, g_labels_s)
        print(f"  Rank-1 accuracy : {r1  * 100:.1f}%")
        print(f"  Rank-5 accuracy : {r5  * 100:.1f}%")
        print(f"  mAP             : {map_s * 100:.1f}%")
        return r1, r5, map_s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_root',   default='dataset/test')
    parser.add_argument('--gallery_root', default='dataset/gallery')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--compare_metrics', action='store_true')
    args = parser.parse_args()

    ckpts = [
        ('checkpoints/model_triplet.pth',       128,  'resnet18', 'Base — Triplet Loss',     False),
        ('checkpoints/model_contrastive.pth',   128,  'resnet18', 'Base — Contrastive Loss', False),
        ('checkpoints/model1_hard_triplet.pth', 256,  'resnet50', 'Model 1 — Hard Triplet (ResNet50)', True),
    ]

    for ckpt, dim, bb, label, cmp in ckpts:
        if os.path.exists(ckpt):
            evaluate(ckpt, args.query_root, args.gallery_root,
                     embedding_dim=dim, backbone=bb, label=label, compare_metrics=cmp)
