import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader

from dataset import GalleryDataset, EVAL_TRANSFORMS
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
    return np.vstack(embeddings), np.array(labels), paths, dataset.label_map


def align_labels(query_lmap, gallery_lmap, query_labels, gallery_labels):
    """Map both label sets to shared identity names so they can be compared."""
    q_int2name = {v: k for k, v in query_lmap.items()}
    g_int2name = {v: k for k, v in gallery_lmap.items()}
    shared = {}
    for name in set(q_int2name.values()) | set(g_int2name.values()):
        shared.setdefault(name, len(shared))
    q_shared = np.array([shared[q_int2name[l]] for l in query_labels])
    g_shared = np.array([shared[g_int2name[l]] for l in gallery_labels])
    return q_shared, g_shared


def show_retrieval(model, query_root, gallery_root, model_name, out_path, top_k=5):
    q_embs, q_labels, q_paths, q_lmap = extract_embeddings(model, query_root)
    g_embs, g_labels, g_paths, g_lmap = extract_embeddings(model, gallery_root)

    q_labels_s, g_labels_s = align_labels(q_lmap, g_lmap, q_labels, g_labels)

    # cosine similarity
    sim = q_embs @ g_embs.T   # (Q, G)
    ranked = np.argsort(-sim, axis=1)  # (Q, G) descending

    n_queries = len(q_paths)
    cols = top_k + 1   # query col + top_k matches
    fig, axes = plt.subplots(n_queries, cols,
                             figsize=(cols * 2.2, n_queries * 2.4))
    if n_queries == 1:
        axes = [axes]

    fig.suptitle(f'Retrieval Results — {model_name}\n'
                 f'Left: Query image | Green border = correct | Red border = wrong',
                 fontsize=12, fontweight='bold')

    for row, (qpath, qlabel) in enumerate(zip(q_paths, q_labels_s)):
        ax = axes[row][0]
        ax.imshow(Image.open(qpath).convert('RGB'))
        ax.set_title(f'Query\nID {os.path.basename(os.path.dirname(qpath))}',
                     fontsize=8, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

        top_indices = ranked[row][:top_k]
        for col, idx in enumerate(top_indices, start=1):
            ax = axes[row][col]
            gpath = g_paths[idx]
            glabel = g_labels_s[idx]
            correct = (glabel == qlabel)

            ax.imshow(Image.open(gpath).convert('RGB'))
            color = 'green' if correct else 'red'
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            sim_score = sim[row, idx]
            ax.set_title(f'Rank {col}\nID {os.path.basename(os.path.dirname(gpath))}\n'
                         f'sim={sim_score:.2f}',
                         fontsize=7,
                         color='green' if correct else 'red')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_root', default='dataset/test')
    parser.add_argument('--gallery_root', default='dataset/gallery')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    for ckpt, name, out in [
        ('checkpoints/model_triplet.pth', 'Triplet Loss', 'retrieval_triplet.png'),
        ('checkpoints/model_contrastive.pth', 'Contrastive Loss', 'retrieval_contrastive.png'),
    ]:
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt} — run train.py first.")
            continue
        model = get_model(embedding_dim=args.embedding_dim, pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        show_retrieval(model, args.query_root, args.gallery_root,
                       name, out, top_k=args.top_k)
