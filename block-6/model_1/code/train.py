import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import TripletDataset, PairDataset, GalleryDataset, PKSampler, TRAIN_TRANSFORMS_V2
from model import get_model
from losses import TripletLoss, ContrastiveLoss, BatchHardTripletLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _save(model, history, save_path):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    loss_path = save_path.replace('.pth', '_losses.json')
    with open(loss_path, 'w') as f:
        json.dump(history, f)
    print(f"  Saved -> {save_path}")
    print(f"  Saved -> {loss_path}")


# ── Base model functions ────────────────────────────────────────────────────

def train_triplet(train_root, save_path, epochs=30, lr=1e-4, batch_size=16, embedding_dim=128):
    dataset = TripletDataset(train_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = get_model(embedding_dim=embedding_dim, pretrained=True, backbone='resnet18').to(DEVICE)
    criterion = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = []
    print(f"\n--- Training with TRIPLET loss (ResNet18) on {DEVICE} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for anchor, positive, negative, _ in loader:
            emb_a = model(anchor.to(DEVICE))
            emb_p = model(positive.to(DEVICE))
            emb_n = model(negative.to(DEVICE))
            loss = criterion(emb_a, emb_p, emb_n)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        history.append(avg)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    _save(model, history, save_path)
    return model


def train_contrastive(train_root, save_path, epochs=30, lr=1e-4, batch_size=16, embedding_dim=128):
    dataset = PairDataset(train_root, pairs_per_sample=4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = get_model(embedding_dim=embedding_dim, pretrained=True, backbone='resnet18').to(DEVICE)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = []
    print(f"\n--- Training with CONTRASTIVE loss (ResNet18) on {DEVICE} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for img1, img2, labels in loader:
            emb1 = model(img1.to(DEVICE))
            emb2 = model(img2.to(DEVICE))
            loss = criterion(emb1, emb2, labels.to(DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        history.append(avg)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    _save(model, history, save_path)
    return model


# ── Model 1: ResNet50 + hard mining + cosine annealing + better augmentation ─

def train_hard_triplet(train_root, save_path, epochs=35, lr=1e-4,
                       embedding_dim=256, P=5, K=4, num_batches=20):
    """
    BatchHard triplet loss with:
    - ResNet50 pretrained backbone
    - PKSampler: P identities × K images per batch
    - Cosine annealing LR
    - Stronger augmentation (random erasing, grayscale)
    - Weight decay to prevent overfitting (loss→0 collapse)
    """
    dataset = GalleryDataset(train_root, transform=TRAIN_TRANSFORMS_V2)
    labels = [lbl for _, lbl, _ in [dataset[i] for i in range(len(dataset))]]
    sampler = PKSampler(labels, P=P, K=K, num_batches=num_batches)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    model = get_model(embedding_dim=embedding_dim, pretrained=True, backbone='resnet50').to(DEVICE)
    criterion = BatchHardTripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    print(f"\n--- Training with BATCH-HARD TRIPLET loss (ResNet50) on {DEVICE} ---")
    print(f"    P={P} identities × K={K} images = batch size {P*K}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls, _ in loader:
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)
            embs = model(imgs)
            loss = criterion(embs, lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        history.append(avg)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    _save(model, history, save_path)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='dataset/train')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--loss', choices=['triplet', 'contrastive', 'both', 'hard'], default='both')
    args = parser.parse_args()

    if args.loss in ('triplet', 'both'):
        train_triplet(args.train_root, 'checkpoints/model_triplet.pth',
                      args.epochs, args.lr, args.batch_size, args.embedding_dim)

    if args.loss in ('contrastive', 'both'):
        train_contrastive(args.train_root, 'checkpoints/model_contrastive.pth',
                          args.epochs, args.lr, args.batch_size, args.embedding_dim)

    if args.loss == 'hard':
        train_hard_triplet(args.train_root, 'checkpoints/model1_hard_triplet.pth',
                           epochs=40, embedding_dim=256)
