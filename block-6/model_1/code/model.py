import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """CNN backbone with a custom L2-normalized embedding head."""

    def __init__(self, embedding_dim=128, pretrained=True, backbone='resnet18'):
        super().__init__()

        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            feature_dim = 512

        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove classifier
        self.embedding = nn.Linear(feature_dim, embedding_dim)

    def forward(self, x, normalize=True):
        features = self.backbone(x).flatten(1)   # (B, feature_dim)
        emb = self.embedding(features)           # (B, embedding_dim)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb


def get_model(embedding_dim=128, pretrained=True, backbone='resnet18'):
    return EmbeddingNet(embedding_dim=embedding_dim, pretrained=pretrained, backbone=backbone)
