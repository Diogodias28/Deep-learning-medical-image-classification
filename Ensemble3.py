import torch
import numpy as np
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
from collections import Counter
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from Dataloader import get_dataloaders, CLASS_NAMES

CONFIG = {
    "data_dir": "dataset",
    "batch_size": 8,
    "num_workers": 0,
    
    "path_a": "checkpoints/best_model_v6.pth",      # convnext-small-mixup-v2
    "path_b": "checkpoints/best_model_v5.pth",      # convnext-small-v5
    "path_c": "checkpoints/convnext-base-flips.pth" # convnext-base-flips
}

def build_model(arch="small", num_classes=4, dropout=0.3):
    if arch == "base":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        in_features = 1024
    else:
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        in_features = 768
        
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    _, _, test_loader, class_names, _ = get_dataloaders(
        data_dir=CONFIG["data_dir"], batch_size=CONFIG["batch_size"]
    )
    
    print("\nA carregar Modelo A (Small - V6)...")
    model_a = build_model(arch="small", num_classes=4).to(device)
    ckpt_a = torch.load(CONFIG["path_a"], map_location=device, weights_only=False)
    model_a.load_state_dict(ckpt_a["model_state"])
    model_a.eval()

    print("A carregar Modelo B (Small - V5)...")
    model_b = build_model(arch="small", num_classes=4).to(device)
    ckpt_b = torch.load(CONFIG["path_b"], map_location=device, weights_only=False)
    model_b.load_state_dict(ckpt_b["model_state"])
    model_b.eval()
    
    print("A carregar Modelo C (Base - Flips)...")
    model_c = build_model(arch="base", num_classes=4).to(device)
    ckpt_c = torch.load(CONFIG["path_c"], map_location=device, weights_only=False)
    model_c.load_state_dict(ckpt_c["model_state"])
    model_c.eval()

    all_preds_a, all_preds_b, all_preds_c = [], [], []
    all_preds_ensemble = []
    all_labels = []

    print("\nA iniciar votação (Hard Majority Voting)...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Inferência"):
            imgs = batch["image"].to(device)
            labels = batch["label"].numpy()
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits_a = model_a(imgs)
                logits_b = model_b(imgs)
                logits_c = model_c(imgs)
            
            preds_a = logits_a.argmax(dim=1).cpu().numpy()
            preds_b = logits_b.argmax(dim=1).cpu().numpy()
            preds_c = logits_c.argmax(dim=1).cpu().numpy()
            
            all_preds_a.extend(preds_a)
            all_preds_b.extend(preds_b)
            all_preds_c.extend(preds_c)
            all_labels.extend(labels)
            
            for pa, pb, pc in zip(preds_a, preds_b, preds_c):
                votos = [pa, pb, pc]
                contagem = Counter(votos)
                
                classe_vencedora, num_votos = contagem.most_common(1)[0]
                
                if num_votos == 1: 
                    final_pred = pa # O Modelo A (V6) atua como juiz de desempate
                else:
                    final_pred = classe_vencedora # Ganha a classe que teve 2 ou 3 votos
                    
                all_preds_ensemble.append(final_pred)

    f1_a = f1_score(all_labels, all_preds_a, average="macro", zero_division=0)
    f1_b = f1_score(all_labels, all_preds_b, average="macro", zero_division=0)
    f1_c = f1_score(all_labels, all_preds_c, average="macro", zero_division=0)
    f1_ens = f1_score(all_labels, all_preds_ensemble, average="macro", zero_division=0)

    print("\n" + "═"*64)
    print(f"  Resultados Individuais:")
    print(f"  Modelo A (V6)   : {f1_a:.4f}")
    print(f"  Modelo B (V5)   : {f1_b:.4f}")
    print(f"  Modelo C (Base) : {f1_c:.4f}")
    print("─"*64)
    print(f"  F1-Score Macro (Hard Voting) : {f1_ens:.4f}")
    print("═"*64)
    
    print("\nRelatório do Ensemble:")
    print(classification_report(all_labels, all_preds_ensemble, target_names=class_names, zero_division=0))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(all_labels, all_preds_ensemble))
    print("═"*64)

if __name__ == "__main__":
    main()