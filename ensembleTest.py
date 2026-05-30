import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from Dataloader import get_dataloaders, CLASS_NAMES

CONFIG = {
    "data_dir"      : "dataset",
    "batch_size"    : 8,
    "num_workers"   : 0,

    "checkpoint_a"  : "checkpoints/best_model_v6.pth",  # convnext-small-mixup-v2
    "checkpoint_b"  : "checkpoints/best_convnext_v4.pth",  # convnext-small-v4

    "weight_a"      : 0.65,
    "weight_b"      : 0.35,

    "num_classes"   : 4,
    "dropout"       : 0.3,
}

BASELINE = 0.738  # F1-Macro do paper


def build_model(num_classes: int = 4, dropout: float = 0.3) -> nn.Module:
    model = models.convnext_small(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(path: str, device: torch.device, label: str) -> nn.Module:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = build_model(CONFIG["num_classes"], CONFIG["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    epoch  = ckpt.get("epoch", "?")
    val_f1 = ckpt.get("val_f1", float("nan"))
    epoch_str = f"{epoch + 1}" if isinstance(epoch, int) else str(epoch)
    print(f"    [{label}] {path}")
    print(f"     └─ guardado na época {epoch_str} | val F1: {val_f1:.4f}")
    return model

@torch.no_grad()
def collect_probs(
    model_a: nn.Module,
    model_b: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_pa, all_pb, all_labels = [], [], []
    use_amp = device.type == "cuda"

    pbar = tqdm(loader, desc="  Inferência", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for batch in pbar:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long()

        with torch.amp.autocast("cuda", enabled=use_amp):
            pa = torch.softmax(model_a(imgs), dim=1)
            pb = torch.softmax(model_b(imgs), dim=1)

        all_pa.append(pa.cpu())
        all_pb.append(pb.cpu())
        all_labels.extend(labels.numpy())

    return (
        torch.cat(all_pa, dim=0).numpy(),
        torch.cat(all_pb, dim=0).numpy(),
        np.array(all_labels),
    )

def report(preds: np.ndarray, labels: np.ndarray,
           class_names: list, title: str) -> float:
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    delta = f1 - BASELINE
    flag  = "V" if delta > 0 else "X"

    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"  F1-Macro : {f1:.4f}  |  delta baseline: {delta:+.4f}  {flag}")
    print(f"{'═'*64}")
    print(classification_report(labels, preds,
                                 target_names=class_names, zero_division=0))
    print("Matriz de Confusão:")
    cm = confusion_matrix(labels, preds)
    col_w = 14
    header = "".join(f"{c[:12]:>{col_w}}" for c in class_names)
    print(f"{'':>20}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"{class_names[i]:>20}{row_str}")
    print(f"{'═'*64}\n")
    return f1


def bl_stats(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Calcula TP, FP, FN para Biliary_Leaks (idx=0)."""
    tp = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 0) & (labels != 0)).sum())
    fn = int(((preds != 0) & (labels == 0)).sum())
    total = int((labels == 0).sum())
    rec   = tp / total if total > 0 else 0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"TP": tp, "FP": fp, "FN": fn, "Total": total,
            "Recall": rec, "Precision": prec, "F1": f1}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"   GPU : {torch.cuda.get_device_name(0)}")

    _, _, test_loader, class_names, _ = get_dataloaders(
        data_dir   =CONFIG["data_dir"],
        batch_size =CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    print("\nA carregar checkpoints...")
    model_a = load_model(CONFIG["checkpoint_a"], device, label="A  (v6)")
    model_b = load_model(CONFIG["checkpoint_b"], device, label="B  (v5)")

    print("\nA correr inferência...")
    probs_a, probs_b, labels = collect_probs(
        model_a, model_b, test_loader, device
    )

    preds_a   = probs_a.argmax(axis=1)
    preds_b   = probs_b.argmax(axis=1)

    wa, wb    = CONFIG["weight_a"], CONFIG["weight_b"]
    probs_ens = wa * probs_a + wb * probs_b
    preds_ens = probs_ens.argmax(axis=1)

    # ── Reports individuais ───────────────────────────────────
    f1_a = report(preds_a, labels, class_names,
                  f"Modelo A (v6) — {CONFIG['checkpoint_a']}")
    f1_b = report(preds_b, labels, class_names,
                  f"Modelo B (v5) — {CONFIG['checkpoint_b']}")
    f1_e = report(preds_ens, labels, class_names,
                  f"Ensemble Soft Voting  (A×{wa} + B×{wb})")

    # ── Biliary_Leak ──────────────────────────────
    bl_a = bl_stats(preds_a,   labels)
    bl_b = bl_stats(preds_b,   labels)
    bl_e = bl_stats(preds_ens, labels)

    print(" Biliary_Leaks — comparação detalhada:")
    print(f"{'':>16} {'TP':>4} {'FP':>4} {'FN':>4}  {'Recall':>7}  {'Prec':>6}  {'F1-BL':>6}")
    print("─" * 60)
    for tag, bl in [("Modelo A (v6)", bl_a), ("Modelo B (v5)", bl_b),
                    ("Ensemble", bl_e)]:
        print(f"  {tag:>14} {bl['TP']:>4} {bl['FP']:>4} {bl['FN']:>4}"
              f"  {bl['Recall']:>7.3f}  {bl['Precision']:>6.3f}  {bl['F1']:>6.3f}")
    print()

    best_ind = max(f1_a, f1_b)
    gain     = f1_e - best_ind
    print("═" * 64)
    print("  SUMÁRIO FINAL")
    print("═" * 64)
    print(f"  Modelo A (v6)       : {f1_a:.4f}")
    print(f"  Modelo B (v5)       : {f1_b:.4f}")
    print(f"  Ensemble {wa}/{wb}       : {f1_e:.4f}  {'V' if gain > 0 else 'X sem ganho'}")
    print(f"  Ganho vs individual : {gain:+.4f}")
    print(f"  Baseline paper      : {BASELINE:.4f}")
    print(f"  Delta baseline      : {f1_e - BASELINE:+.4f}  "
          f"{'BATIDO!' if f1_e > BASELINE else 'Não Batido'}")
    print("═" * 64)

if __name__ == "__main__":
    main()