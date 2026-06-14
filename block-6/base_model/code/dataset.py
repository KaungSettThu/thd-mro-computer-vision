import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_folder(root):
    """Return list of (image_path, label_int) from root/identity_id/img.jpg structure."""
    samples = []
    label_map = {}
    for identity in sorted(os.listdir(root)):
        identity_path = os.path.join(root, identity)
        if not os.path.isdir(identity_path):
            continue
        if identity not in label_map:
            label_map[identity] = len(label_map)
        label = label_map[identity]
        for fname in os.listdir(identity_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append((os.path.join(identity_path, fname), label))
    return samples, label_map


class TripletDataset(Dataset):
    """For each anchor, randomly samples a positive and a negative."""

    def __init__(self, root, transform=TRAIN_TRANSFORMS):
        self.transform = transform
        self.samples, self.label_map = load_folder(root)

        # group paths by label
        self.by_label = {}
        for path, label in self.samples:
            self.by_label.setdefault(label, []).append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, anchor_label = self.samples[idx]

        # positive: different image, same identity
        pos_path = random.choice(
            [p for p in self.by_label[anchor_label] if p != anchor_path]
            or self.by_label[anchor_label]
        )

        # negative: random image from a different identity
        neg_label = random.choice(
            [l for l in self.by_label if l != anchor_label]
        )
        neg_path = random.choice(self.by_label[neg_label])

        anchor = self.transform(Image.open(anchor_path).convert('RGB'))
        positive = self.transform(Image.open(pos_path).convert('RGB'))
        negative = self.transform(Image.open(neg_path).convert('RGB'))

        return anchor, positive, negative, anchor_label


class PairDataset(Dataset):
    """Generates pairs (img1, img2, label) for contrastive loss. label=1 means same identity."""

    def __init__(self, root, transform=TRAIN_TRANSFORMS, pairs_per_sample=4):
        self.transform = transform
        self.samples, self.label_map = load_folder(root)
        self.by_label = {}
        for path, label in self.samples:
            self.by_label.setdefault(label, []).append(path)

        self.pairs = self._build_pairs(pairs_per_sample)

    def _build_pairs(self, pairs_per_sample):
        pairs = []
        for path, label in self.samples:
            # positive pairs
            for _ in range(pairs_per_sample // 2):
                pos = random.choice(
                    [p for p in self.by_label[label] if p != path]
                    or self.by_label[label]
                )
                pairs.append((path, pos, 1))
            # negative pairs
            for _ in range(pairs_per_sample // 2):
                neg_label = random.choice([l for l in self.by_label if l != label])
                neg = random.choice(self.by_label[neg_label])
                pairs.append((path, neg, 0))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = self.transform(Image.open(p1).convert('RGB'))
        img2 = self.transform(Image.open(p2).convert('RGB'))
        return img1, img2, label


class GalleryDataset(Dataset):
    """Simple flat dataset returning (image, label, path) for evaluation."""

    def __init__(self, root, transform=EVAL_TRANSFORMS):
        self.transform = transform
        self.samples, self.label_map = load_folder(root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        return img, label, path
