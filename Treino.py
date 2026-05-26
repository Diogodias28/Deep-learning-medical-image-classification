import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

_project_root = os.path.dirname(os.path.abspath(__file__))
_removed_paths = []
for _path in list(sys.path):
    if os.path.abspath(_path or os.getcwd()) == _project_root:
        sys.path.remove(_path)
        _removed_paths.append(_path)

import wandb

for _path in reversed(_removed_paths):
    sys.path.insert(0, _path)

from Dataloader import get_dataloaders, CLASS_NAMES

CONFIG = {
    "data_dir"           : "dataset",
    "batch_size"         : 8,
    "num_workers"        : 0,
    "model_name"         : "convnext-small",
    "num_classes"        : 4,        
    "dropout"            : 0.3,
    "use_amp"            : True,

    "focal_gamma"        : 2.0,
    "label_smoothing"    : 0.1,

    "epochs"             : 45,
    "learning_rate"      : 5e-5,
    "weight_decay"       : 3e-2,
    "grad_clip"          : 1.0,
    "early_stop_patience": 15,
    
    "mixup_prob"         : 0.5,
    "cutmix_prob"        : 0.2,
    "mixup_alpha"        : 0.4,
    "cutmix_alpha"       : 1.0,
    
    "swa_start_epoch"    : 15,
    "swa_lr"             : 1e-5,

    "T_max"              : 45,

    "checkpoint_dir"     : "checkpoints",
    "best_model_name"    : "best_model_v6.pth",
    "wandb_project"      : "cpre-miqr-classification",
    "wandb_run_name"     : "convnext-small-final",
}

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_p = torch.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        if targets.dim() == 1:
            soft = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        else:
            soft = targets

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            soft = soft * (1.0 - self.label_smoothing) + smooth

        focal_weight = torch.pow(1 - p, self.gamma)
        loss = -focal_weight * soft * log_p
        
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)
            
        return loss.sum(dim=1).mean() if self.reduction == "mean" else loss.sum()


def build_model(num_classes: int = 4, dropout: float = 0.5) -> nn.Module:
    model = models.convnext_small(
        weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    total  = sum(p.numel() for p in model.parameters())
    treina = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConvNeXt-Small — total: {total/1e6:.1f}M | treináveis: {treina/1e6:.1f}M")
    return model

def _to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(
        labels.size(0), num_classes, device=labels.device
    ).scatter_(1, labels.unsqueeze(1), 1.0)


def mixup_batch(imgs, labels, num_classes, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    imgs_mix  = lam * imgs + (1.0 - lam) * imgs[idx]
    soft_a = _to_onehot(labels, num_classes)
    soft_b = _to_onehot(labels[idx], num_classes)
    soft_mix = lam * soft_a + (1.0 - lam) * soft_b
    return imgs_mix, soft_mix


def cutmix_batch(imgs, labels, num_classes, alpha=1.0, max_area_ratio=0.25):
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - max_area_ratio)
    batch_size, _, H, W = imgs.shape
    idx = torch.randperm(batch_size, device=imgs.device)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    imgs_mix = imgs.clone()
    imgs_mix[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam_real = 1.0 - (float((x2 - x1) * (y2 - y1)) / float(H * W))
    soft_a   = _to_onehot(labels, num_classes)
    soft_b   = _to_onehot(labels[idx], num_classes)
    soft_mix = lam_real * soft_a + (1.0 - lam_real) * soft_b
    return imgs_mix, soft_mix


def train_epoch(
    model, loader, optimizer, scheduler, criterion, device, grad_clip, scaler,
    num_classes=4, mixup_prob=0.5, cutmix_prob=0.2,
    mixup_alpha=0.4, cutmix_alpha=1.0,
):
    model.train()
    total_loss = 0.0
    all_preds, all_labels_hard = [], []

    pbar = tqdm(loader, desc="  🔁 Train", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for batch in pbar:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)

        r = random.random()
        augmentation_used = "none"

        if r < mixup_prob:
            imgs, soft_labels = mixup_batch(imgs, labels, num_classes, mixup_alpha)
            augmentation_used = "mixup"
        elif r < mixup_prob + cutmix_prob:
            imgs, soft_labels = cutmix_batch(imgs, labels, num_classes, cutmix_alpha)
            augmentation_used = "cutmix"
        else:
            soft_labels = _to_onehot(labels, num_classes)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(imgs)
            loss   = criterion(logits, soft_labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            hard_approx = soft_labels.argmax(dim=1).cpu().numpy()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels_hard.extend(hard_approx)

        pbar.set_postfix(loss=f"{loss.item():.4f}", aug=augmentation_used[0].upper())

    scheduler.step()

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels_hard, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for batch in pbar:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, np.array(all_preds), np.array(all_labels)

@torch.no_grad()
def eval_epoch_tta(model, loader, device, num_classes=4, n_augments=5):
    model.eval()
    all_preds, all_labels = [], []

    def _zoom(x, factor=1.05):
        _, _, H, W = x.shape
        new_H = int(H / factor)
        new_W = int(W / factor)
        y1 = (H - new_H) // 2
        x1 = (W - new_W) // 2
        cropped = x[:, :, y1:y1+new_H, x1:x1+new_W]
        return torch.nn.functional.interpolate(
            cropped, size=(H, W), mode="bilinear", align_corners=False
        )

    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.clamp(x + 0.1, -3, 3),
        lambda x: torch.clamp(x - 0.1, -3, 3),
        lambda x: _zoom(x, factor=1.05),
    ]

    pbar = tqdm(loader, desc="  🔍 TTA  ", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for batch in pbar:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long()

        probs_sum = torch.zeros(imgs.size(0), num_classes, device=device)

        for aug in tta_transforms[:n_augments]:
            imgs_aug = aug(imgs)
            with torch.amp.autocast("cuda"):
                logits = model(imgs_aug)
            probs_sum += torch.softmax(logits, dim=1)

        preds = (probs_sum / n_augments).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return f1, np.array(all_preds), np.array(all_labels)


def evaluate_test(model, test_loader, criterion, device, class_names, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"checkpoint = torch.load(checkpoint_path, map_location=device)")
    print(f"\n Teste — checkpoint epoch {checkpoint['epoch']+1} "
          f"(Val F1: {checkpoint['val_f1']:.4f})")

    test_f1, preds, labels = eval_epoch_tta(model, test_loader, device)

    print("\n" + "═" * 60)
    print(f"  F1-Score Macro (Teste): {test_f1:.4f}")
    print("═" * 60)
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
    cm = confusion_matrix(labels, preds)
    print("Matriz de Confusão:")
    print(cm)
    print("═" * 60 + "\n")

    wandb.log({
        "test/f1_macro"        : test_f1,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=labels.tolist(),
            preds=preds.tolist(), class_names=class_names,
        ),
    })
    return test_f1


def main(cfg: dict):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}")
    if device.type == "cuda":
        print(f"   GPU : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / cfg["best_model_name"]

    wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)

    data_dir = Path(cfg["data_dir"])
    if not data_dir.exists():
        fallback = Path(__file__).resolve().parent / "dataset"
        if fallback.exists():
            print(f" A usar '{fallback}' em vez de '{data_dir}'.")
            data_dir = fallback
        else:
            raise FileNotFoundError(f"data_dir '{data_dir}' não encontrado.")

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    model = build_model(
        num_classes=cfg["num_classes"],
        dropout    =cfg["dropout"],
    ).to(device)

    criterion = FocalLoss(
        gamma           = cfg["focal_gamma"],
        label_smoothing = cfg["label_smoothing"],
        alpha           = class_weights.to(device),
    )
    print(f"\n📉 FocalLoss(gamma={cfg['focal_gamma']}, "
        f"label_smoothing={cfg['label_smoothing']}, alpha={class_weights.tolist()})")

    optimizer = AdamW(
        model.parameters(),
        lr          =cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        eps         =1e-8,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T_max"], eta_min=1e-6)
    print(f" CosineAnnealingLR | T_max={cfg['T_max']}")

    use_amp = cfg.get("use_amp", False) and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    print(f" AMP (mixed precision): {'activado' if use_amp else 'desactivado'}\n")

    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(
        optimizer,
        swa_lr          = cfg["swa_lr"],
        anneal_epochs   = 5,
        anneal_strategy = "cos",
    )
    swa_started       = False
    best_val_f1       = 0.0
    patience_counter  = 0

    print(" A iniciar treino...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train F1':>8} | "
          f"{'Val Loss':>8} | {'Val F1':>7} | {'LR':>10} | Status")
    print("─" * 84)

    for epoch in range(cfg["epochs"]):

        if epoch >= cfg["swa_start_epoch"] and not swa_started:
            swa_started = True
            print(f"\n🔄 SWA activado no epoch {epoch+1} "
                  f"(LR: {cfg['learning_rate']:.1e} → {cfg['swa_lr']:.1e})\n")

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, cfg["grad_clip"], scaler,
            num_classes  = cfg["num_classes"],
            mixup_prob   = cfg["mixup_prob"],
            cutmix_prob  = cfg["cutmix_prob"],
            mixup_alpha  = cfg["mixup_alpha"],
            cutmix_alpha = cfg["cutmix_alpha"],
        )

        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        status = ""

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_f1": best_val_f1, "config": cfg},
                best_model_path,
            )
            status = "💾 best"
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"\n⏹️  Early stopping no epoch {epoch+1}.")
                break

        swa_flag = "🔄" if swa_started else "  "
        print(
            f"{epoch+1:>6} | {train_loss:>10.4f} | {train_f1:>8.4f} | "
            f"{val_loss:>8.4f} | {val_f1:>7.4f} | {current_lr:>10.2e} | {swa_flag} {status}"
        )

        wandb.log({
            "epoch": epoch+1, "train/loss": train_loss,
            "train/f1_macro": train_f1, "val/loss": val_loss,
            "val/f1_macro": val_f1, "val/best_f1": best_val_f1,
            "train/lr": current_lr, "swa/active": int(swa_started),
            "early_stop/patience": patience_counter,
        })

    if swa_started:
        print("\n🔄 A actualizar BatchNorm do modelo SWA...")
        update_bn(train_loader, swa_model, device=device)

        swa_model_path = ckpt_dir / "swa_model_v5.pth"
        torch.save(
            {"epoch": epoch, "model_state": swa_model.module.state_dict(),
             "val_f1": best_val_f1, "config": cfg},
            swa_model_path,
        )
        print(f"💾 Modelo SWA guardado em {swa_model_path}")

    print("\n" + "═"*84)
    print("  AVALIAÇÃO FINAL — MODELO NORMAL (melhor val F1)")
    test_f1_normal = evaluate_test(
        model, test_loader, criterion, device,
        class_names, str(best_model_path),
    )

    if swa_started:
        print("\n" + "═"*84)
        print("  AVALIAÇÃO FINAL — MODELO SWA")
        test_f1_swa = evaluate_test(
            swa_model.module, test_loader, criterion, device,
            class_names, str(swa_model_path),
        )
        wandb.run.summary["test_f1_swa"] = test_f1_swa
        print(f"\n Normal: {test_f1_normal:.4f} | SWA: {test_f1_swa:.4f} "
              f"| Ganho SWA: {test_f1_swa - test_f1_normal:+.4f}")

    test_f1 = max(test_f1_normal,
                  test_f1_swa if swa_started else 0)
    wandb.run.summary["best_val_f1"] = best_val_f1
    wandb.run.summary["test_f1"]     = test_f1
    wandb.finish()

    print(f"\n - Treino concluído.")
    print(f"   Melhor Val F1 : {best_val_f1:.4f}")
    print(f"   Test F1 Macro : {test_f1:.4f}")
    print(f"   Checkpoint    : {best_model_path}")


# ══════════════════════════════════════════════════════════════
# 8. ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino CPRE MIQR-CC v3")
    parser.add_argument("--data_dir",   type=str,   default=CONFIG["data_dir"])
    parser.add_argument("--batch_size", type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--epochs",     type=int,   default=CONFIG["epochs"])
    parser.add_argument("--lr",         type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--run_name",   type=str,   default=CONFIG["wandb_run_name"])
    args = parser.parse_args()

    CONFIG["data_dir"]       = args.data_dir
    CONFIG["batch_size"]     = args.batch_size
    CONFIG["epochs"]         = args.epochs
    CONFIG["learning_rate"]  = args.lr
    CONFIG["wandb_run_name"] = args.run_name
    CONFIG["mixup_prob"]    = 0.5
    CONFIG["cutmix_prob"]   = 0.2
    CONFIG["mixup_alpha"]   = 0.4
    CONFIG["cutmix_alpha"]  = 1.0

    main(CONFIG)