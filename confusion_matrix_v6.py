import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from Dataloader import CLASS_NAMES, get_dataloaders
from Treino import build_model


def _predict_plain(model, loader, device):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Avaliação", leave=False):
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def _predict_tta(model, loader, device, n_augments=5):
    def _zoom(x, factor=1.05):
        _, _, h, w = x.shape
        new_h = int(h / factor)
        new_w = int(w / factor)
        y1 = (h - new_h) // 2
        x1 = (w - new_w) // 2
        cropped = x[:, :, y1:y1 + new_h, x1:x1 + new_w]
        return torch.nn.functional.interpolate(
            cropped, size=(h, w), mode="bilinear", align_corners=False
        )

    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.clamp(x + 0.1, -3, 3),
        lambda x: torch.clamp(x - 0.1, -3, 3),
        lambda x: _zoom(x, factor=1.05),
    ]

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Avaliação TTA", leave=False):
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)

            probs_sum = torch.zeros(imgs.size(0), len(CLASS_NAMES), device=device)

            for aug in tta_transforms[:n_augments]:
                imgs_aug = aug(imgs)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs_aug)
                probs_sum += torch.softmax(logits, dim=1)

            preds = (probs_sum / n_augments).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def _save_confusion_matrix(cm, class_names, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=220)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.6,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=15, pad=12)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Gera uma matriz de confusão em PNG para o checkpoint best_model_v6.pth."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model_v6.pth",
        help="Caminho para o checkpoint .pth.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Pasta raiz do dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="confusion_matrix_v6.png",
        help="Ficheiro PNG de saída.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size para avaliação.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usa TTA na previsão final.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    model = build_model(
        num_classes=cfg.get("num_classes", len(CLASS_NAMES)),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Checkpoint carregado: epoch {checkpoint.get('epoch', '?')}")
    print(f"Melhor Val F1 gravado: {checkpoint.get('val_f1', 'n/a')}")

    _, _, test_loader, class_names, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )

    if args.tta:
        y_true, y_pred = _predict_tta(model, test_loader, device)
    else:
        y_true, y_pred = _predict_plain(model, test_loader, device)

    cm = confusion_matrix(y_true, y_pred)
    print("\n" + classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Matriz de confusão:\n", cm)

    output_path = Path(args.output)
    title = f"Confusion Matrix - best_model_v6 ({'TTA' if args.tta else 'plain'})"
    _save_confusion_matrix(cm, class_names, output_path, title)
    print(f"PNG guardado em: {output_path}")


if __name__ == "__main__":
    main()