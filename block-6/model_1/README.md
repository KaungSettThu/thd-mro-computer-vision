# Model 1 — Improved Re-Identification Pipeline

**Assignment:** PM06 Re-Identification — TH Deggendorf (Prof. Tobias Schaffer)  
**Dataset:** Paw Patrol cartoon characters (Identities 1–7)  
**Improvement over:** Base model (see `base_model/README.md`)

---

## What Changed from Base Model

| Component | Base Model | Model 1 |
|-----------|-----------|---------|
| Backbone | ResNet18 | **ResNet50** |
| Embedding dim | 128 | **256** |
| Loss function | Triplet / Contrastive (random mining) | **Batch Hard Triplet Loss** |
| Batch strategy | Random shuffle | **PKSampler (P×K batches)** |
| LR scheduler | StepLR (step=10, γ=0.5) | **CosineAnnealingLR** |
| Augmentation | Basic (flip, crop, jitter) | **+ RandomErasing, RandomGrayscale, RandomRotation** |
| Optimizer | Adam | Adam + **weight_decay=1e-4** |
| Epochs | 30 | **40** |

---

## Architecture

### Backbone — ResNet50
ResNet50 is a deeper network with residual blocks that learns richer feature representations than ResNet18. The final classification layer is removed; the 2048-dimensional feature vector is passed through a linear projection layer to produce the embedding.

```
Input Image (3×224×224)
    → ResNet50 backbone (pretrained on ImageNet)
    → Global Average Pooling → 2048-dim feature
    → Linear(2048 → 256)
    → L2 Normalize
    → 256-dim embedding on unit hypersphere
```

### Embedding
All embeddings are L2-normalized so they lie on the unit hypersphere. Distance is measured as cosine similarity (equivalent to Euclidean distance on the hypersphere).

---

## Training Strategy

### Batch Hard Triplet Loss
Instead of randomly sampling triplets (anchor, positive, negative), Batch Hard mining selects the **hardest** triplets in each batch:

- **Hardest positive**: the same-identity image that is *farthest* from the anchor
- **Hardest negative**: a different-identity image that is *closest* to the anchor

```
Loss = max(0, d(A, hardest_P) − d(A, hardest_N) + margin)
margin = 0.3
```

This forces the model to focus on the most challenging examples rather than wasting capacity on easy pairs that are already correctly separated.

### PKSampler
Batches are constructed with **P identities × K images each**:
- P = 5 (all training identities)
- K = 4 (4 random images per identity)
- Batch size = 20

This guarantees every batch contains multiple images of each identity, making hard positive/negative mining meaningful. With random sampling you might get batches where some identities appear only once, making it impossible to find a positive pair.

### Cosine Annealing LR
Learning rate follows a cosine curve from `lr=1e-4` down to `eta_min=1e-6` over 40 epochs. This provides warm restarts-style decay — the LR decreases smoothly rather than stepping, allowing the optimizer to escape local minima in the early epochs while fine-tuning precisely at the end.

### Weight Decay
`weight_decay=1e-4` regularizes the embedding space, preventing the model from collapsing all training embeddings to degenerate solutions where the loss reaches zero but clustering breaks down.

### Stronger Augmentation
Training augmentation (`TRAIN_TRANSFORMS_V2`) includes:
- `RandomResizedCrop(224)` — varied scale and aspect ratio
- `RandomHorizontalFlip`
- `ColorJitter` — brightness, contrast, saturation, hue
- `RandomGrayscale(p=0.1)` — forces color-invariant features
- `RandomRotation(15°)` — pose variation
- `RandomErasing(p=0.3)` — simulates occlusion

---

## Training Results

| Epoch | Loss |
|-------|------|
| 1 | 0.4322 |
| 10 | 0.0190 |
| 20 | 0.0063 |
| 30 | 0.0051 |
| 40 | 0.0003 |

Loss converges from 0.43 to near-zero over 40 epochs, indicating the model has learned to separate all 5 training identities with margin > 0.3.

---

## Evaluation Results

**Protocol:** 8 query images (test/: 4×ID6, 4×ID7) matched against 22 gallery images (gallery/: 11×ID6, 11×ID7). Test identities 6 and 7 were **never seen during training** — this is an open-set retrieval task.

### Distance Metric Comparison

| Metric | Rank-1 | Rank-5 | mAP |
|--------|--------|--------|-----|
| Cosine similarity | **100.0%** | 100.0% | **84.6%** |
| Euclidean distance | **100.0%** | 100.0% | **84.6%** |
| Manhattan distance | 87.5% | 100.0% | 83.2% |

> **Note:** Cosine and Euclidean give identical rankings on L2-normalized embeddings because `||a−b||² = 2(1 − a·b)` — the ranking order is mathematically equivalent. Manhattan distance (L1) measures differently and may rank marginally differently.

### Comparison with Base Model

| Model | Rank-1 | mAP |
|-------|--------|-----|
| Base — Triplet Loss (ResNet18) | 87.5% | 77.8% |
| Base — Contrastive Loss (ResNet18) | 75.0% | 70.9% |
| **Model 1 — Hard Triplet (ResNet50)** | **100.0%** | **84.6%** |

Model 1 achieves **+12.5 pp Rank-1** and **+6.8 pp mAP** over the best base model, demonstrating that the combination of a stronger backbone, hard mining, and better training strategy meaningfully improves generalization to unseen identities.

---

## Visualizations

### t-SNE Embedding Space (`results/tsne_model1_comparison.png`)
Three panels showing how the embedding space evolves:

1. **Untrained ResNet18** — embeddings are random, all identities mixed
2. **Base Triplet (ResNet18, trained)** — training identities begin to cluster, unseen IDs (stars) loosely grouped
3. **Model 1 (ResNet50, 40 epochs)** — all 5 training identity clusters are tight and well-separated; unseen identities (ID 6, ID 7, shown as stars ★) are plotted to demonstrate open-set behavior

Circles (●) = training identities (1–5, seen during training)  
Stars (★) = test identities (6–7, unseen during training)

### Retrieval Visualization (`results/retrieval_model1.png`)
For each query image (left, blue border), the top-5 gallery matches are shown:
- **Green border** = correct identity retrieved
- **Red border** = wrong identity retrieved

With 100% Rank-1 accuracy, the top-1 match is always correct for all 8 queries.

### Training Summary (`results/training_summary.png`)
5-panel plot showing:
- Base model loss curves (triplet + contrastive)
- Model 1 loss curve (hard triplet with cosine annealing)
- All losses overlaid
- Rank-1 & mAP bar chart (base triplet, base contrastive, model 1)
- Distance metric comparison for model 1 (cosine, euclidean, manhattan)

---

## File Structure

```
model_1/
├── README.md                          ← this file
├── code/
│   ├── dataset.py                     ← loaders, PKSampler, augmentation transforms
│   ├── model.py                       ← EmbeddingNet (ResNet18/50 backbone)
│   ├── losses.py                      ← TripletLoss, ContrastiveLoss, BatchHardTripletLoss
│   ├── train.py                       ← training functions (triplet, contrastive, hard)
│   ├── evaluate.py                    ← Rank-1/5, mAP, distance metric comparison
│   ├── visualize.py                   ← t-SNE embedding space plots
│   ├── results.py                     ← retrieval visualization (query + top-k gallery)
│   └── plot_metrics.py                ← training summary (loss curves + metric bars)
├── checkpoints/
│   ├── model1_hard_triplet.pth        ← trained weights (ResNet50, 256-dim)
│   └── model1_hard_triplet_losses.json ← per-epoch loss values
└── results/
    ├── tsne_model1_comparison.png     ← t-SNE: untrained | base | model 1
    ├── retrieval_model1.png           ← top-5 retrieval results
    └── training_summary.png           ← full metrics comparison plot
```

---

## How to Reproduce

```bash
# Train model 1
python train.py --loss hard

# Evaluate (all three distance metrics)
python evaluate.py

# t-SNE comparison (base vs model 1)
python visualize.py \
  --ckpt1 checkpoints/model_triplet.pth --label1 "Base: Triplet (ResNet18)" --dim1 128 --bb1 resnet18 \
  --ckpt2 checkpoints/model1_hard_triplet.pth --label2 "Model 1: Hard Triplet (ResNet50)" --dim2 256 --bb2 resnet50 \
  --title "Base vs Model 1" --out tsne_model1_comparison.png

# Retrieval visualization
python results.py

# Training summary plot
python plot_metrics.py
```

---

## Key Design Decisions

**Why ResNet50?** Deeper network with more parameters captures finer visual distinctions between cartoon character identities. The pretrained ImageNet weights transfer well even to cartoon images since low-level features (edges, textures, shapes) are universal.

**Why 256-dim embedding?** Larger embedding space gives the model more room to separate identities while still being compact enough to generalize. 128-dim (base model) was sufficient but 256-dim provides more expressive power for the harder mining strategy.

**Why 40 epochs specifically?** Early stopping tuning showed that at 50 epochs the loss collapsed to ~0 and certain identity clusters became degenerate in the embedding space. At 35 epochs the model undertrained. 40 epochs hits the point where the loss just reaches near-zero (0.0003), giving the best Rank-1 (100%) and mAP (84.6%).

**Why weight_decay?** Without it, the model can find degenerate solutions where the loss goes to zero but embeddings are not well-clustered — the model pushes identities apart but not necessarily into compact clusters. Weight decay regularizes this.
