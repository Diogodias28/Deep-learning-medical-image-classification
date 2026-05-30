import os
import glob
from collections import Counter
import cv2
import numpy as np
import torch
from monai.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    RepeatChanneld, RandRotated, RandFlipd, RandZoomd,
    RandAdjustContrastd, RandShiftIntensityd,
    NormalizeIntensityd, ToTensord, MapTransform,
)

CLASS_NAMES  = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMG_SIZE     = 512

class CLAHEd(MapTransform):
    def __init__(self, keys, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__(keys)
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data):
        d     = dict(data)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                 tileGridSize=self.tile_grid_size)
        for key in self.keys:
            img    = d[key]
            img_np = img.numpy() if isinstance(img, torch.Tensor) else img
            img_2d = img_np[0]
            img_u8    = cv2.normalize(img_2d, None, 0, 255,
                                      cv2.NORM_MINMAX).astype(np.uint8)
            img_clahe = clahe.apply(img_u8).astype(np.float32) / 255.0
            d[key]    = np.expand_dims(img_clahe, axis=0)
        return d


def _build_train_transforms():
    return Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(IMG_SIZE, IMG_SIZE), mode="bilinear"),
        CLAHEd(keys=["image"], clip_limit=2.0),
        RepeatChanneld(keys=["image"], repeats=3),

        RandRotated(keys=["image"], range_x=0.26, prob=0.6,
                    keep_size=True, mode="bilinear"),
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        RandZoomd(keys=["image"], min_zoom=0.85, max_zoom=1.15,
                  prob=0.5, keep_size=True),
        NormalizeIntensityd(keys=["image"], channel_wise=True),
        ToTensord(keys=["image"]),
    ])

def _build_val_test_transforms():
    return Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(IMG_SIZE, IMG_SIZE), mode="bilinear"),
        CLAHEd(keys=["image"], clip_limit=2.0),
        RepeatChanneld(keys=["image"], repeats=3),
        NormalizeIntensityd(keys=["image"], channel_wise=True),
        ToTensord(keys=["image"]),
    ])

def _load_split(data_dir, split):
    split_dir   = os.path.join(data_dir, split)
    image_paths = glob.glob(os.path.join(split_dir, "*", "*.*"))
    
    data_list = []
    for p in image_paths:
        original_class = os.path.basename(os.path.dirname(p))
        if original_class in CLASS_TO_IDX:
            if original_class == "Biliary_Leaks":
                binary_label = 1 # Positivo
            else:
                binary_label = 0 # Negativo (Outros) 
            data_list.append({"image": p, "label": binary_label})
    return data_list

def _build_sampler(labels):
    counts  = Counter(labels)
    n_total = len(labels)
    weights = torch.tensor(
        [n_total / (len(CLASS_NAMES) * counts[l]) for l in labels],
        dtype=torch.float32,
    )
    print("\n WeightedRandomSampler:")
    for i, name in enumerate(CLASS_NAMES):
        cnt = counts.get(i, 0)
        print(f"   {name:20s} → {cnt:4d} imgs | peso = "
              f"{n_total/(len(CLASS_NAMES)*max(cnt,1)):.4f}")
    return WeightedRandomSampler(weights, num_samples=n_total, replacement=True)

def _compute_class_weights(labels):
    counts  = Counter(labels)
    n_total = len(labels)
    weights = torch.tensor(
        [n_total / (len(CLASS_NAMES) * counts.get(i, 1))
         for i in range(len(CLASS_NAMES))],
        dtype=torch.float32,
    )
    print(f"\n Class weights: {weights.tolist()}")
    return weights

def get_dataloaders(data_dir, batch_size=8, num_workers=0):
    train_data = _load_split(data_dir, "train")
    val_data   = _load_split(data_dir, "val")
    test_data  = _load_split(data_dir, "test")

    print(f"train: {len(train_data)} | val: {len(val_data)} | test: {len(test_data)}")

    train_labels  = [d["label"] for d in train_data]
    sampler       = _build_sampler(train_labels)
    class_weights = _compute_class_weights(train_labels)

    train_loader = DataLoader(
        Dataset(data=train_data, transform=_build_train_transforms()),
        batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Dataset(data=val_data, transform=_build_val_test_transforms()),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        Dataset(data=test_data, transform=_build_val_test_transforms()),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, CLASS_NAMES, class_weights