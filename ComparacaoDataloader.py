import torch
import numpy as np
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from Dataloader import get_dataloaders

CONFIG = {
    "data_dir": "dataset",
    "batch_size": 8,
    "num_workers": 0,
    "path_v6": "checkpoints/best_model_v6.pth"
}

OLD_RESULTS = {
    "macro_f1": 0.6780,
    "Biliary_Leaks_f1": 0.4000,
    "Lithiasis_f1": 0.8192,
    "Normal_f1": 0.6800,
    "Stricture_f1": 0.8129,
    "Biliary_Leaks_tp": 6
}

def build_model(num_classes=4, dropout=0.3):
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    in_features = 768
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    _, _, test_loader, class_names, _ = get_dataloaders(
        data_dir=CONFIG["data_dir"], batch_size=CONFIG["batch_size"]
    )
    
    print("\nA carregar modelo V6...")
    model = build_model(num_classes=4).to(device)
    ckpt = torch.load(CONFIG["path_v6"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds = []
    all_labels = []

    print("A executar inferencia...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Progresso"):
            imgs = batch["image"].to(device)
            labels = batch["label"].numpy()
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    current_macro_f1 = report_dict["macro avg"]["f1-score"]
    current_biliary_f1 = report_dict["Biliary_Leaks"]["f1-score"]
    current_lithiasis_f1 = report_dict["Lithiasis"]["f1-score"]
    current_normal_f1 = report_dict["Normal"]["f1-score"]
    current_stricture_f1 = report_dict["Stricture"]["f1-score"]
    current_biliary_tp = cm[0][0] # Index 0 e Biliary_Leaks

    print("\n" + "="*60)
    print("RELATORIO DE COMPARACAO: DATALOADER ANTIGO VS ATUAL")
    print("="*60)
    
    print(f"{'Metrica':<20} | {'Antigo (W&B)':<15} | {'Atual':<10} | {'Diferenca'}")
    print("-" * 60)
    
    def print_comparison(name, old_val, new_val, is_integer=False):
        diff = new_val - old_val
        if is_integer:
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"{name:<20} | {old_val:<15} | {new_val:<10} | {diff_str}")
        else:
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            print(f"{name:<20} | {old_val:<15.4f} | {new_val:<10.4f} | {diff_str}")

    print_comparison("F1-Macro Global", OLD_RESULTS["macro_f1"], current_macro_f1)
    print_comparison("F1 Biliary_Leaks", OLD_RESULTS["Biliary_Leaks_f1"], current_biliary_f1)
    print_comparison("F1 Lithiasis", OLD_RESULTS["Lithiasis_f1"], current_lithiasis_f1)
    print_comparison("F1 Normal", OLD_RESULTS["Normal_f1"], current_normal_f1)
    print_comparison("F1 Stricture", OLD_RESULTS["Stricture_f1"], current_stricture_f1)
    print("-" * 60)
    print_comparison("Biliary_Leaks (TP)", OLD_RESULTS["Biliary_Leaks_tp"], current_biliary_tp, is_integer=True)
    
    print("\nMatriz de Confusao Atual:")
    print(cm)
    print("="*60)

if __name__ == "__main__":
    main()