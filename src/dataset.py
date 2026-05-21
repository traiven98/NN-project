import os
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

from config import (
    CSV_PATH, IMG_DIRS, IMG_SIZE, BATCH_SIZE, SEED,
    CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD,
)


def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    img_path_map = {}
    for img_dir in IMG_DIRS:
        if not os.path.exists(img_dir):
            continue
        for fname in os.listdir(img_dir):
            img_id = os.path.splitext(fname)[0]
            img_path_map[img_id] = os.path.join(img_dir, fname)

    df['path'] = df['image_id'].map(img_path_map)
    df = df.dropna(subset=['path']).reset_index(drop=True)
    df['label'] = df['dx'].map({c: i for i, c in enumerate(CLASS_NAMES)})

    print(f"Loaded {len(df)} images across {df['dx'].nunique()} classes")
    return df


def split_data(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=SEED
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, val_transform


class SkinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row['label'])


def build_loaders(train_df, val_df, test_df):
    train_transform, val_transform = get_transforms()

    train_dataset = SkinDataset(train_df, transform=train_transform)
    val_dataset   = SkinDataset(val_df,   transform=val_transform)
    test_dataset  = SkinDataset(test_df,  transform=val_transform)

    label_counts   = Counter(train_df['label'].values)
    class_weights  = {cls: 1.0 / count for cls, count in label_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_df['label'].values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,    num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, val_transform
