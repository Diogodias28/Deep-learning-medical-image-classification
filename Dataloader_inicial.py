import os
import glob
from collections import Counter

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    CenterSpatialCropd,
    RepeatChanneld,
    NormalizeIntensityd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    EnsureTyped,
    ToTensord,
)
from monai.transforms import MapTransform


IMG_SIZE    = 512
IMG_RESIZE  = 560

CLASS_NAMES  = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

class ApplyCLAHE(MapTransform):
    def __init__(self, keys, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        super().__init__(keys)
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                 tileGridSize=self.tile_grid_size)
        for key in self.keys:
            img    = d[key]
            img_np = img[0].numpy() if hasattr(img, "numpy") else np.array(img[0])
            lo, hi = img_np.min(), img_np.max()
            if hi - lo > 1e-6:
                img_u8 = ((img_np - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                img_u8 = np.zeros_like(img_np, dtype=np.uint8)
            img_clahe = clahe.apply(img_u8)
            d[key]    = img_clahe.astype(np.float32)[np.newaxis, ...] / 255.0
        return d


def _build_train_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(IMG_RESIZE, IMG_RESIZE), mode="bilinear"),
        CenterSpatialCropd(keys=["image"], roi_size=(IMG_SIZE, IMG_SIZE)),

        ApplyCLAHE(keys=["image"], clip_limit=2.0, tile_grid_size=(8, 8)),

        RepeatChanneld(keys=["image"], repeats=3),

        NormalizeIntensityd(keys=["image"], channel_wise=True),

        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.2),
        RandRotated(
            keys=["image"],
            range_x=0.26,      # ±15°
            prob=0.6,
            keep_size=True,
            mode="bilinear",
        ),
        RandZoomd(
            keys=["image"],
            min_zoom=0.85,
            max_zoom=1.15,
            prob=0.5,
            keep_size=True,
        ),

        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.5)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.4),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.03),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            prob=0.2,
        ),

        EnsureTyped(keys=["image"], dtype=torch.float32),
        ToTensord(keys=["image"]),
    ])


def _build_val_test_transforms() -> Compose:
    """Determinístico — sem augmentation."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(IMG_RESIZE, IMG_RESIZE), mode="bilinear"),
        CenterSpatialCropd(keys=["image"], roi_size=(IMG_SIZE, IMG_SIZE)),
        ApplyCLAHE(keys=["image"], clip_limit=2.0, tile_grid_size=(8, 8)),
        RepeatChanneld(keys=["image"], repeats=3),
        # Mesma normalização que no treino: por-imagem
        NormalizeIntensityd(keys=["image"], channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        ToTensord(keys=["image"]),
    ])

def _load_split(data_dir: str, split: str) -> list[dict]:
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"  Pasta {split_dir} não encontrada!")

    image_paths = glob.glob(os.path.join(split_dir, "*", "*.*"))
    data_list   = []
    for img_path in image_paths:
        class_name = os.path.basename(os.path.dirname(img_path))
        if class_name in CLASS_TO_IDX:
            data_list.append({
                "image": img_path,
                "label": CLASS_TO_IDX[class_name],
            })
    return data_list

def _build_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts    = Counter(labels)
    n_total   = len(labels)
    n_classes = len(CLASS_NAMES)

    print("\n WeightedRandomSampler — distribuição do treino:")
    class_w = {}
    for cls_idx in range(n_classes):
        cnt           = counts.get(cls_idx, 1)
        w             = n_total / (n_classes * cnt)
        class_w[cls_idx] = w
        print(f"   {CLASS_NAMES[cls_idx]:20s} → {cnt:4d} amostras | peso = {w:.4f}")

    sample_weights = torch.tensor(
        [class_w[lbl] for lbl in labels], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )

def _compute_class_weights(labels: list[int]) -> torch.Tensor:
    counts    = Counter(labels)
    n_total   = len(labels)
    n_classes = len(CLASS_NAMES)
    weights   = torch.tensor(
        [n_total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    )
    print(f"\nClass weights para a Loss: {weights.tolist()}")
    return weights


def get_dataloaders(
    data_dir:    str,
    batch_size:  int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], torch.Tensor]:
    train_data = _load_split(data_dir, "train")
    val_data   = _load_split(data_dir, "val")
    test_data  = _load_split(data_dir, "test")

    print(f" Split 'train': {len(train_data)} imagens | {len(CLASS_NAMES)} classes")
    print(f" Split 'val'  : {len(val_data)} imagens  | {len(CLASS_NAMES)} classes")
    print(f" Split 'test' : {len(test_data)} imagens  | {len(CLASS_NAMES)} classes")

    train_dataset = Dataset(data=train_data, transform=_build_train_transforms())
    val_dataset   = Dataset(data=val_data,   transform=_build_val_test_transforms())
    test_dataset  = Dataset(data=test_data,  transform=_build_val_test_transforms())

    train_labels  = [item["label"] for item in train_data]
    sampler       = _build_sampler(train_labels)
    class_weights = _compute_class_weights(train_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, CLASS_NAMES, class_weights


def verify_pipeline(dataset, n: int = 4, save_path: str = "verificacao_pipeline.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        sample  = dataset[i]
        img     = sample["image"]
        img_np  = img[0].numpy()

        lo, hi  = img_np.min(), img_np.max()
        img_vis = (img_np - lo) / (hi - lo + 1e-8)

        axes[0, i].imshow(img_vis, cmap="gray")
        axes[0, i].set_title(f"Label: {CLASS_NAMES[sample['label']]}")
        axes[0, i].axis("off")
        axes[1, i].hist(img_np.flatten(), bins=50, color="steelblue")
        axes[1, i].set_title("Intensidades")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Guardada em '{save_path}'")

    img = dataset[0]["image"]
    print(f"   Shape: {img.shape} | dtype: {img.dtype}")
    print(f"   Min: {img.min():.3f} | Max: {img.max():.3f} | Mean: {img.mean():.3f}")


if __name__ == "__main__":
    import sys
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "dataset"

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        data_dir=DATA_DIR, batch_size=8,
    )
    print(f"\nBatches treino : {len(train_loader)}")
    print(f"Batches val    : {len(val_loader)}")
    print(f"Batches teste  : {len(test_loader)}")

    batch = next(iter(train_loader))
    print(f"\nBatch OK → images: {batch['image'].shape} | labels: {batch['label']}")