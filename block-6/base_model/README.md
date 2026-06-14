# PM06 Re-Identification — Base Model

**Course:** Image Processing & Computer Vision — TH Deggendorf, Campus Cham  
**Assignment:** Block 6 — Re-Identification  

---

## Overview

This is the base model implementation for the re-identification assignment. The goal is to determine whether two images show the same identity using a learned embedding space, without classification — the model never predicts a fixed class label. Instead it maps every image to a vector such that same-identity images are close and different-identity images are far apart.

---

## Dataset

All images were captured manually for this assignment.

```
dataset/
  train/          ← identities 1–5  (seen during training)
    1/            ← 15 images
    2/            ← 15 images
    3/            ← 15 images
    4/            ← 15 images
    5/            ← 15 images
  gallery/        ← identities 6–7  (never seen during training)
    6/            ← 11 images
    7/            ← 11 images
  test/           ← query images for identities 6–7
    6/            ← 4 images
    7/            ← 4 images
```

**Key point:** Identities 6 and 7 are completely held out — they never appear in training. This makes it an open-set problem, where the model must generalize to identities it has never seen.

Each identity was photographed from:
- Different viewpoints
- Different distances
- Different lighting conditions

---

## Model Architecture

**Backbone:** ResNet18 pretrained on ImageNet  
**Embedding head:** Single fully-connected layer (512 → 128 dimensions)  
**Output:** L2-normalized 128-dimensional embedding vector

```
Input Image (128×128)
       ↓
ResNet18 backbone (pretrained, frozen feature extraction)
       ↓
Flatten → 512-dim feature vector
       ↓
Linear layer → 128-dim embedding
       ↓
L2 Normalize → unit vector on 128-dim sphere
```

The L2 normalization ensures all embeddings lie on a unit hypersphere, making cosine similarity equivalent to dot product — fast and numerically stable.

---

## Training

Two separate models were trained, one for each loss function.

### Model A — Triplet Loss

**Loss function:**

```
L = max(0, d(A, P) − d(A, N) + margin)
```

Where:
- A = anchor image (e.g. identity 1, image a)
- P = positive image (same identity, different image)
- N = negative image (different identity)
- d(·,·) = cosine distance
- margin = 0.3

Each training step feeds 3 images. The loss pulls the anchor closer to its positive and pushes it away from the negative by at least the margin. If the condition is already satisfied the loss is zero — the model only learns from hard examples.

**Training config:**
- Epochs: 30
- Learning rate: 1e-4 (halved every 10 epochs)
- Batch size: 16
- Data augmentation: random horizontal flip, color jitter

**Loss curve:**
| Epoch | Loss |
|---|---|
| 1 | 0.3008 |
| 10 | 0.0516 |
| 20 | 0.0056 |
| 30 | 0.0000 |

---

### Model B — Contrastive Loss

**Loss function:**

```
L = y · D² + (1−y) · max(0, margin − D)²
```

Where:
- y = 1 for same identity pair, y = 0 for different
- D = Euclidean distance between embeddings
- margin = 1.0

Each training step feeds 2 images + a binary same/different label. Positive pairs are pulled together; negative pairs are pushed apart until they exceed the margin.

**Loss curve:**
| Epoch | Loss |
|---|---|
| 1 | 0.3317 |
| 10 | 0.0410 |
| 20 | 0.0213 |
| 30 | 0.0122 |

Contrastive loss converges more slowly than triplet loss and does not reach zero, because pair-based learning provides less structured gradient signal than triplet comparisons.

---

## Evaluation

### Protocol

- **Query images:** `dataset/test/` — 4 images per identity × 2 identities = 8 queries
- **Gallery images:** `dataset/gallery/` — 11 images per identity × 2 identities = 22 gallery items

For each query image:
1. Extract its 128-dim embedding using the trained model
2. Compute cosine similarity against all 22 gallery embeddings
3. Rank gallery images from highest to lowest similarity
4. Check if the correct identity appears at rank 1 (Rank-1) or within top k (Rank-k)

Both query and gallery identities (6 and 7) were **never seen during training**.

### Results

| Metric | Triplet Loss | Contrastive Loss |
|---|---|---|
| Rank-1 accuracy | **75.0%** | **75.0%** |
| Rank-5 accuracy | **100.0%** | **100.0%** |
| mAP | **77.1%** | **67.5%** |

**Interpretation:**
- **Rank-1 75%** — for 6 out of 8 queries, the top retrieved gallery image belongs to the correct identity
- **Rank-5 100%** — every query finds its correct identity within the top 5 results
- **Triplet > Contrastive on mAP** — triplet loss directly optimizes the ranking geometry, so the correct matches are ranked higher on average
- The results are achieved on **unseen identities**, confirming open-set generalization

---

## t-SNE Visualization

All 97 images (75 train + 22 gallery) are embedded and projected to 2D using t-SNE.

**Plot 1 — Untrained (Random Weights):**  
All identities scattered randomly. No structure, no clustering. This is the baseline with no learning.

**Plot 2 — Triplet Loss:**  
Training identities (circles, 1–5) form clear tight clusters. Unseen identities (stars, 6–7) show partial grouping — the model generalizes but not perfectly.

**Plot 3 — Contrastive Loss:**  
Similar clustering of training identities. Unseen identities again partially grouped.

The fact that unseen identities (6, 7) form *some* clustering despite never appearing in training proves the model learned a general similarity representation, not identity memorization. This is the core open-set property.

---

## File Structure

```
base_model/
  README.md             ← this file
  code/
    dataset.py          ← TripletDataset, PairDataset, GalleryDataset loaders
    model.py            ← ResNet18 + embedding head
    losses.py           ← TripletLoss, ContrastiveLoss
    train.py            ← training loop for both loss functions
    evaluate.py         ← Rank-1, Rank-5, mAP evaluation
    visualize.py        ← t-SNE comparison plot (all 7 identities, 3 models)
    results.py          ← retrieval visualization (query + top-5 ranked matches)
  checkpoints/
    model_triplet.pth       ← saved weights, triplet loss model
    model_contrastive.pth   ← saved weights, contrastive loss model
  results/
    tsne_comparison.png         ← t-SNE: untrained vs triplet vs contrastive
    retrieval_triplet.png       ← query→gallery retrieval grid, triplet model
    retrieval_contrastive.png   ← query→gallery retrieval grid, contrastive model
```

---

## How to Run

**1. Install dependencies**
```bash
pip install torch torchvision numpy Pillow scikit-learn matplotlib
```

**2. Train both models**
```bash
python code/train.py --epochs 30 --loss both
```

**3. Evaluate (Rank-1, Rank-5, mAP)**
```bash
python code/evaluate.py
```

**4. t-SNE visualization**
```bash
python code/visualize.py
```

**5. Retrieval result grid**
```bash
python code/results.py
```

---

## Limitations of This Base Model

- **No bounding box annotations:** Images are fed full-size. If an image has a large background, the model may learn background features rather than identity-specific ones.
- **Small dataset:** 75 training images across 5 identities is minimal. Larger datasets typically improve embedding quality significantly.
- **CPU training:** No GPU used, so training is slower but fully functional.
- **No hard negative mining:** Triplets are sampled randomly. Online hard mining (selecting the most informative negatives) would improve learning efficiency.
