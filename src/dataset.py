"""
dataset.py
==========
PyTorch Dataset for Wardrobe Circle clothing classification.
"""

import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224  # EfficientNet-B0 / ResNet-18 standard input size


def get_train_transform() -> transforms.Compose:
    """Light augmentation for training — nothing that changes clothing identity."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for validation and test."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────
class WardrobeDataset(Dataset):
    """
    Parameters
    ----------
    csv_path:   path to train.csv / val.csv / test.csv
    class_names: sorted list of wardrobe category names
    transform:  torchvision transform
    """

    def __init__(
        self,
        csv_path: str,
        class_names: list[str],
        transform: transforms.Compose,
    ) -> None:
        self.df = pd.read_csv(csv_path, dtype=str)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform

        # Filter to known classes and existing images
        self.df = self.df[
            self.df["wardrobe_category"].isin(self.class_to_idx)
        ].copy()
        self.df = self.df[
            self.df["image_path"].apply(os.path.exists)
        ].copy()
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path: str = row["image_path"]
        label: int = self.class_to_idx[row["wardrobe_category"]]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a blank image if the file is corrupt
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label
