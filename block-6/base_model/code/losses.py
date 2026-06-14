import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Standard triplet loss with a margin.
    Embeddings must already be L2-normalized.
    Uses cosine distance (1 - cosine_similarity) since embeddings are normalized,
    which is equivalent to squared Euclidean up to a constant.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # cosine similarity on unit vectors == dot product
        d_pos = 1.0 - (anchor * positive).sum(dim=1)   # (B,)
        d_neg = 1.0 - (anchor * negative).sum(dim=1)   # (B,)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pairs.
    label=1 means same identity (positive), label=0 means different (negative).
    L = label * D^2 + (1 - label) * max(0, margin - D)^2
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        # Euclidean distance between L2-normalized embeddings is in [0, 2]
        dist = F.pairwise_distance(emb1, emb2, p=2)
        labels = labels.float()
        loss_pos = labels * dist.pow(2)
        loss_neg = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (loss_pos + loss_neg).mean()
