import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision import models
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import wandb

from DataLoaderEnsemble import get_dataloaders, CLASS_NAMES


CONFIG = {
    "data_dir"           : "dataset",
    "batch_size"         : 8,
    "num_workers"        : 0,

    "model_name"         : "convnext-small-Billiary",
    "num_classes"        : 2,
    "dropout"            : 0.3,
    "use_amp"            : True,

    "focal_gamma"        : 2.0,
    "label_smoothing"    : 0.1,

    "epochs"             : 45,
    "learning_rate"      : 5e-5,
    "weight_decay"       : 3e-2,
    "grad_clip"          : 1.0,
    "early_stop_patience": 15,
    "T_max"              : 45,

    "checkpoint_dir"     : "checkpoints",
    "best_model_name"    : "billiary_leaks_spec.pth",
    "wandb_project"      : "cpre-miqr-classification",
    "wandb_run_name"     : "convnext-small-Billiary",
}


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ── MODELO ────────────────────────────────────────────────────
def build_model(num_classes=4, dropout=0.3):
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features   # 768
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    total  = sum(p.numel() for p in model.parameters())
    treina = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConvNeXt-Small — total: {total/1e6:.1f}M | treináveis: {treina/1e6:.1f}M")
    return model


# ── FOCAL LOSS ────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight         =self.alpha,
            label_smoothing=self.label_smoothing,
            reduction      ='none',
        )
        pt          = torch.exp(-ce_loss)
        focal_loss  = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ── TREINO ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, device, cfg, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    pbar = tqdm(loader, desc="  Train", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["use_amp"]):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    return total_loss / len(loader), f1_score(all_labels, all_preds,
                                               average="macro", zero_division=0)


# ── VALIDAÇÃO ─────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0))


# ── AVALIAÇÃO FINAL ───────────────────────────────────────────
def evaluate_test(model, test_loader, device, class_names, checkpoint_path, label=""):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n{label} checkpoint epoch {ckpt['epoch']+1} "
          f"(Val F1: {ckpt['val_f1']:.4f})")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  Teste {label}"):
            imgs   = batch["image"].to(device)
            labels = batch["label"]
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    test_f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)

    print("\n" + "═"*60)
    print(f"  F1-Score Binary (Teste {label}): {test_f1:.4f}")
    print("═"*60)
    print(classification_report(all_labels, all_preds,
                                 target_names=class_names, zero_division=0))
    print("Matriz de Confusão:")
    print(confusion_matrix(all_labels, all_preds))
    print("═"*60 + "\n")

    wandb.log({
        f"test{label}/f1_binary"      : test_f1,
        f"test{label}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=all_labels,
            preds=all_preds, class_names=class_names,
        ),
    })
    return test_f1


# ── MAIN ──────────────────────────────────────────────────────
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"   GPU : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["wandb_run_name"],
               config=CONFIG)

    ckpt_dir = Path(CONFIG["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / CONFIG["best_model_name"]

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        data_dir=CONFIG["data_dir"], batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    model = build_model(num_classes=CONFIG["num_classes"],
                        dropout=CONFIG["dropout"]).to(device)

    criterion = FocalLoss(
        gamma          =CONFIG["focal_gamma"],
        alpha          =None,
        label_smoothing=CONFIG["label_smoothing"],
    )
    print(f"\nFocalLoss(gamma={CONFIG['focal_gamma']}, "
          f"label_smoothing={CONFIG['label_smoothing']}, alpha=None)")

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                      weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["T_max"], eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda") if CONFIG["use_amp"] and \
                device.type == "cuda" else None

    print(f"CosineAnnealingLR | T_max={CONFIG['T_max']} (= epochs → 1 ciclo limpo)")
    print(f"AMP: {'activado' if scaler else 'desactivado'}")

    best_val_f1, patience_ctr = 0.0, 0

    print("A iniciar treino...\n")
    print(f"{'Ep':>4} | {'TrLoss':>8} | {'TrF1':>6} | {'VlLoss':>8} | "
          f"{'VlF1':>6} | {'LR':>9} | Status")
    print("─" * 72)

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, CONFIG, scaler,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        val_loss, val_f1 = eval_epoch(model, val_loader, criterion, device)
        status = ""

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_ctr = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_f1": best_val_f1, "config": CONFIG}, best_path)
            status = "-> best"
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["early_stop_patience"]:
                print(f"\nEarly stopping no epoch {epoch+1}.")
                break

        print(f"{epoch+1:>4} | {train_loss:>8.4f} | {train_f1:>6.4f} | "
              f"{val_loss:>8.4f} | {val_f1:>6.4f} | {current_lr:>9.2e}")

        wandb.log({"epoch": epoch+1, "train/loss": train_loss,
                   "train/f1_binary": train_f1, "val/loss": val_loss,
                   "val/f1_binary": val_f1, "val/best_f1": best_val_f1,
                   "train/lr": current_lr,
                   "early_stop/patience": patience_ctr})

    # ── Avaliação final ────────────────────────────────────────
    print("\n" + "═"*72)
    test_f1_normal = evaluate_test(model, test_loader, device,
                                   class_names, str(best_path), label="Normal")

    best = test_f1_normal
    wandb.run.summary.update({"best_val_f1": best_val_f1, "test_f1": best})
    wandb.finish()

    print(f"\nTreino concluído.")
    print(f"   Melhor Val F1  : {best_val_f1:.4f}")
    print(f"   Test F1 Normal : {test_f1_normal:.4f}")
    delta = best - 0.738
    print(f"   Melhor Test F1 : {best:.4f}  |  baseline delta: {delta:+.4f} "
          f"{'BATIDO!' if delta > 0 else 'Não Batido'}")


if __name__ == "__main__":
    main()