import os
import random
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Stronger augmentation for model_1 — simulates occlusion, lighting, viewpoint variation
TRAIN_TRANSFORMS_V2 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
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


class PKSampler(Sampler):
    """
    Samples P identities × K images per batch for hard negative mining.
    Yields multiple batches per epoch by repeating num_batches times.
    """
    def __init__(self, labels, P, K, num_batches=20):
        self.by_label = {}
        for idx, lbl in enumerate(labels):
            self.by_label.setdefault(lbl, []).append(idx)
        self.P = min(P, len(self.by_label))
        self.K = K
        self.num_batches = num_batches
        self.label_list = list(self.by_label.keys())

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            chosen = random.sample(self.label_list, self.P)
            for lbl in chosen:
                batch.extend(random.choices(self.by_label[lbl], k=self.K))
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


class TripletDataset(Dataset):
    """For each anchor, randomly samples a positive and a negative."""

    def __init__(self, root, transform=TRAIN_TRANSFORMS):
        self.transform = transform
        self.samples, self.label_map = load_folder(root)
        self.by_label = {}
        for path, label in self.samples:
            self.by_label.setdefault(label, []).append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, anchor_label = self.samples[idx]
        pos_path = random.choice(
            [p for p in self.by_label[anchor_label] if p != anchor_path]
            or self.by_label[anchor_label]
        )
        neg_label = random.choice([l for l in self.by_label if l != anchor_label])
        neg_path = random.choice(self.by_label[neg_label])

        anchor   = self.transform(Image.open(anchor_path).convert('RGB'))
        positive = self.transform(Image.open(pos_path).convert('RGB'))
        negative = self.transform(Image.open(neg_path).convert('RGB'))
        return anchor, positive, negative, anchor_label


class PairDataset(Dataset):
    """Generates (img1, img2, label) pairs for contrastive loss."""

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
            for _ in range(pairs_per_sample // 2):
                pos = random.choice(
                    [p for p in self.by_label[label] if p != path]
                    or self.by_label[label]
                )
                pairs.append((path, pos, 1))
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
    """Simple flat dataset returning (image, label, path) for evaluation or hard mining."""

    def __init__(self, root, transform=EVAL_TRANSFORMS):
        self.transform = transform
        self.samples, self.label_map = load_folder(root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        return img, label, path
