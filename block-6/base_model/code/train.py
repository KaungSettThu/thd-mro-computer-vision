import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import TripletDataset, PairDataset
from model import get_model
from losses import TripletLoss, ContrastiveLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_triplet(train_root, save_path, epochs=30, lr=1e-4, batch_size=16, embedding_dim=128):
    dataset = TripletDataset(train_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = get_model(embedding_dim=embedding_dim, pretrained=True).to(DEVICE)
    criterion = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = []
    print(f"\n--- Training with TRIPLET loss on {DEVICE} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for anchor, positive, negative, _ in loader:
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        history.append(avg)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    loss_path = save_path.replace('.pth', '_losses.json')
    with open(loss_path, 'w') as f:
        json.dump(history, f)
    print(f"  Saved -> {save_path}")
    print(f"  Saved -> {loss_path}")
    return model


def train_contrastive(train_root, save_path, epochs=30, lr=1e-4, batch_size=16, embedding_dim=128):
    dataset = PairDataset(train_root, pairs_per_sample=4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = get_model(embedding_dim=embedding_dim, pretrained=True).to(DEVICE)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = []
    print(f"\n--- Training with CONTRASTIVE loss on {DEVICE} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for img1, img2, labels in loader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            labels = labels.to(DEVICE)

            emb1 = model(img1)
            emb2 = model(img2)

            loss = criterion(emb1, emb2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        history.append(avg)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    loss_path = save_path.replace('.pth', '_losses.json')
    with open(loss_path, 'w') as f:
        json.dump(history, f)
    print(f"  Saved -> {save_path}")
    print(f"  Saved -> {loss_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='dataset/train')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--loss', choices=['triplet', 'contrastive', 'both'], default='both')
    args = parser.parse_args()

    if args.loss in ('triplet', 'both'):
        train_triplet(
            args.train_root,
            save_path='checkpoints/model_triplet.pth',
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
        )

    if args.loss in ('contrastive', 'both'):
        train_contrastive(
            args.train_root,
            save_path='checkpoints/model_contrastive.pth',
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
        )
