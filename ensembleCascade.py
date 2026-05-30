import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from Dataloader import get_dataloaders, CLASS_NAMES
from Treino import build_model

CONFIG = {
    "data_dir": "dataset",
    "batch_size": 8,
    "num_workers": 0,
    
    "path_multi"  : "checkpoints/best_model_v6.pth", # convnext-small-mixup-v2
    "path_binary" : "checkpoints/billiary_leaks_spec.pth", # Modelo Especialista só para Biliary Leaks (Binário: Leak vs No-Leak)
    
    "binary_threshold": 0.25
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    _, _, test_loader, class_names, _ = get_dataloaders(
        data_dir=CONFIG["data_dir"], batch_size=CONFIG["batch_size"]
    )
    
    print("\nA carregar modelo Multiclasse (V6)...")
    model_multi = build_model(num_classes=4, dropout=0.3).to(device)
    ckpt_multi = torch.load(CONFIG["path_multi"], map_location=device, weights_only=False)
    model_multi.load_state_dict(ckpt_multi["model_state"])
    model_multi.eval()
    print(f"   └─ Época: {ckpt_multi.get('epoch', '?')} | Val F1: {ckpt_multi.get('val_f1', '?'):.4f}")

    print("A carregar modelo Especialista Binário...")
    model_binary = build_model(num_classes=2, dropout=0.3).to(device)
    ckpt_binary = torch.load(CONFIG["path_binary"], map_location=device, weights_only=False)
    model_binary.load_state_dict(ckpt_binary["model_state"])
    model_binary.eval()
    print(f"   └─ Época: {ckpt_binary.get('epoch', '?')} | Val F1 (Bin): {ckpt_binary.get('val_f1', '?'):.4f}")

    all_preds = []
    all_labels = []
    
    threshold = CONFIG["binary_threshold"]
    print(f"\nA iniciar Cascata (Hard Cascade) | Threshold Binário = {threshold}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Inferência"):
            imgs = batch["image"].to(device)
            labels = batch["label"].numpy()
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits_multi = model_multi(imgs)
                logits_binary = model_binary(imgs)
                
            probs_multi = torch.softmax(logits_multi, dim=1).cpu().numpy()
            probs_binary = torch.softmax(logits_binary, dim=1).cpu().numpy()
            
            for i in range(len(labels)):
                
                prob_leak_v6 = probs_multi[i, 0]
                prob_leak_bin = probs_binary[i, 1]
                
                peso_v6 = 0.65
                peso_binario = 0.35
                
                nova_prob_leak = (prob_leak_v6 * peso_v6) + (prob_leak_bin * peso_binario)
                
                adj_probs_multi = probs_multi[i].copy()
                adj_probs_multi[0] = nova_prob_leak
                
                final_pred = np.argmax(adj_probs_multi)
                
                all_preds.append(final_pred)
                all_labels.append(labels[i])

    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    print("\n" + "═"*64)
    print(f"  F1-Score Macro (Cascade Ensemble): {test_f1:.4f}")
    print("═"*64)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(all_labels, all_preds))
    print("═"*64)

if __name__ == "__main__":
    main()