import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Standard triplet loss with random triplet sampling. Uses cosine distance."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = 1.0 - (anchor * positive).sum(dim=1)
        d_neg = 1.0 - (anchor * negative).sum(dim=1)
        return F.relu(d_pos - d_neg + self.margin).mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pairs.
    L = label * D^2 + (1 - label) * max(0, margin - D)^2
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        dist = F.pairwise_distance(emb1, emb2, p=2)
        labels = labels.float()
        loss_pos = labels * dist.pow(2)
        loss_neg = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (loss_pos + loss_neg).mean()


class BatchHardTripletLoss(nn.Module):
    """
    Online hard negative mining: for each anchor in the batch,
    selects the hardest positive (most distant same-identity)
    and hardest negative (closest different-identity).
    Works on L2-normalized embeddings using Euclidean distance.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # pairwise Euclidean distance matrix (B, B)
        dist = torch.cdist(embeddings, embeddings, p=2)

        losses = []
        for i in range(len(labels)):
            pos_mask = (labels == labels[i])
            pos_mask[i] = False
            neg_mask = (labels != labels[i])

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            hardest_pos = dist[i][pos_mask].max()
            hardest_neg = dist[i][neg_mask].min()
            losses.append(F.relu(hardest_pos - hardest_neg + self.margin))

        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(losses).mean()
