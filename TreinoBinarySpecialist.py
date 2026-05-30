
"""
TreinoBinarySpecialist.py
─────────────────────────
Treina um classificador binário ResNet-18:
  Classe 1 → Biliary_Leaks
  Classe 0 → Lithiasis | Normal | Stricture

O modelo especialista é usado em INFERÊNCIA conjunta com o ConvNeXt-Small
principal para melhorar o recall de Biliary_Leaks (a classe mais rara).

USO:
  python TreinoBinarySpecialist.py --data_dir dataset

DEPOIS:
  O checkpoint gerado (checkpoints/specialist_biliary.pth) é passado ao
  evaluate_test() do Treino.py através do argumento specialist_ckpt.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from tqdm import tqdm

# ── WandB sem conflito com pasta local ────────────────────────
_project_root = os.path.dirname(os.path.abspath(__file__))
_removed = []
for _p in list(sys.path):
    if os.path.abspath(_p or os.getcwd()) == _project_root:
        sys.path.remove(_p); _removed.append(_p)
import wandb
for _p in reversed(_removed): sys.path.insert(0, _p)

from Dataloader import get_binary_dataloaders, CLASS_NAMES


CONFIG = {
    "data_dir"           : "dataset",
    "batch_size"         : 8,
    "num_workers"        : 0,

    "epochs"             : 40,
    "learning_rate"      : 5e-5,
    "weight_decay"       : 1e-2,
    "grad_clip"          : 1.0,
    "early_stop_patience": 12,
    "T_max"              : 40,
    "use_amp"            : True,

    # Threshold para classificar como Biliary_Leaks no teste conjunto.
    # Optimizado no val — podes ajustar manualmente depois.
    "decision_threshold" : 0.35,

    "checkpoint_dir"     : "checkpoints",
    "model_name"         : "specialist_biliary.pth",
    "wandb_project"      : "cpre-miqr-classification",
    "wandb_run_name"     : "resnet18-binary-biliary-specialist",
}


# ──────────────────────────────────────────────────────────────
# MODELO — ResNet-18 binário
# ──────────────────────────────────────────────────────────────
def build_specialist(dropout=0.3):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features   # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),       # saída: logit único → BCEWithLogitsLoss
    )
    total  = sum(p.numel() for p in model.parameters())
    treina = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️  ResNet-18 Specialist — total: {total/1e6:.1f}M | treináveis: {treina/1e6:.1f}M")
    return model


# ──────────────────────────────────────────────────────────────
# TRAIN EPOCH
# ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    pbar = tqdm(loader, desc="  🔁 Train", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)   # float para BCE

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(imgs).squeeze(1)   # [B]
            loss   = criterion(logits, labels)

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
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend((probs > 0.5).astype(int))
        all_labels.extend(labels.cpu().numpy().astype(int))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return avg_loss, f1


# ──────────────────────────────────────────────────────────────
# EVAL EPOCH
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    pbar = tqdm(loader, desc="  🔍 Val  ", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int))

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > threshold).astype(int)
    f1    = f1_score(all_labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return total_loss / len(loader), f1, auc, all_probs, all_labels


# ──────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION (no val)
# ──────────────────────────────────────────────────────────────
def find_best_threshold(val_probs, val_labels):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.10, 0.70, 0.01):
        preds = (val_probs > t).astype(int)
        f1    = f1_score(val_labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"   Melhor threshold (val): {best_t:.2f}  →  F1 binário = {best_f1:.4f}")
    return best_t


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main(cfg):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")

    ckpt_dir = Path(cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg["model_name"]

    wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)

    train_loader, val_loader, test_loader = get_binary_dataloaders(
        data_dir   =cfg["data_dir"],
        batch_size =cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    model     = build_specialist().to(device)
    # pos_weight compensa ainda mais o desequilíbrio: 957 negativas / 110 positivas ≈ 8.7
    pos_weight = torch.tensor([957.0 / 110.0]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"\n📉 BCEWithLogitsLoss | pos_weight={pos_weight.item():.2f}")

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"],
                      weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T_max"], eta_min=1e-7)
    use_amp   = cfg["use_amp"] and device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_f1, patience_ctr = 0.0, 0
    best_threshold = 0.35

    print("\n🚀 A treinar especialista binário Biliary_Leaks...\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Train F1':>8} | {'Val F1':>7} | {'Val AUC':>7} | Status")
    print("─" * 60)

    for epoch in range(cfg["epochs"]):
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            cfg["grad_clip"], scaler,
        )
        val_loss, val_f1, val_auc, val_probs, val_labels = eval_epoch(
            model, val_loader, criterion, device,
        )

        status = ""
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_ctr = 0
            # Encontra o melhor threshold no val
            best_threshold = find_best_threshold(val_probs, val_labels)
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_f1"     : best_val_f1,
                "threshold"  : best_threshold,
                "config"     : cfg,
            }, best_path)
            status = "💾 best"
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["early_stop_patience"]:
                print(f"\n⏹️  Early stopping no epoch {epoch+1}.")
                break

        print(f"{epoch+1:>6} | {train_loss:>8.4f} | {train_f1:>8.4f} | "
              f"{val_f1:>7.4f} | {val_auc:>7.4f} | {status}")
        wandb.log({"epoch": epoch+1, "train/loss": train_loss, "train/f1": train_f1,
                   "val/f1": val_f1, "val/auc": val_auc, "val/best_f1": best_val_f1})

    # Avaliação final no teste
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    threshold = ckpt["threshold"]
    print(f"\n🔍 A avaliar no teste com threshold={threshold:.2f}...")

    _, test_f1, test_auc, test_probs, test_labels = eval_epoch(
        model, test_loader, criterion, device, threshold=threshold,
    )
    preds = (test_probs > threshold).astype(int)

    print("\n" + "═"*50)
    print(f"  Especialista — Teste F1 binário : {test_f1:.4f}")
    print(f"  Especialista — Teste AUC        : {test_auc:.4f}")
    print("═"*50)
    print(classification_report(test_labels, preds,
                                 target_names=["Não-BL", "Biliary_Leaks"], zero_division=0))

    wandb.run.summary.update({"test_f1": test_f1, "test_auc": test_auc,
                               "best_threshold": threshold})
    wandb.finish()

    print(f"\n✅ Especialista treinado. Checkpoint: {best_path}")
    print(f"   Threshold óptimo (val): {threshold:.2f}")
    print(f"   Usa este checkpoint em Treino.py → evaluate_test(..., specialist_ckpt='{best_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,   default=CONFIG["data_dir"])
    parser.add_argument("--batch_size", type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--epochs",     type=int,   default=CONFIG["epochs"])
    args = parser.parse_args()
    CONFIG.update({"data_dir": args.data_dir, "batch_size": args.batch_size,
                   "epochs": args.epochs})
    main(CONFIG)