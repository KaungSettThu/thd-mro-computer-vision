import torch
import torch.nn as nn
from torchvision import models


class EmbeddingNet(nn.Module):
    """ResNet18 backbone with a custom L2-normalized embedding head."""

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # keep everything except the original classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 512, 1, 1)
        self.embedding = nn.Linear(512, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)          # (B, 512, 1, 1)
        features = features.flatten(1)       # (B, 512)
        emb = self.embedding(features)       # (B, embedding_dim)
        emb = nn.functional.normalize(emb, p=2, dim=1)  # L2 normalize
        return emb


def get_model(embedding_dim=128, pretrained=True):
    return EmbeddingNet(embedding_dim=embedding_dim, pretrained=pretrained)
