import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from Treino import build_model
from Dataloader import get_dataloaders, CLASS_NAMES

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_cam_from_dataloader(loader, model, target_layer, device, save_path="attention_map.png"):
    batch = next(iter(loader))
    
    input_tensor = batch["image"].to(device) 
    label_idx = batch["label"][0].item()
    true_class = CLASS_NAMES[label_idx]
    
    img_vis = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
    
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        pred_idx = logits.argmax(dim=1).item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = torch.nn.functional.softmax(logits, dim=1)[0][pred_idx].item()
    
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    cam_image = show_cam_on_image(img_vis, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(11, 5), dpi=200)
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_vis)
    plt.title(f"Radiografia Original\nClasse Real: {true_class}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"Attention Map da Rede (V6)\nPred: {pred_class} ({confidence*100:.1f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Mapa de atenção guardado com sucesso em: {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    _, _, test_loader, _, _ = get_dataloaders(data_dir="dataset", batch_size=1, num_workers=0)
    
    model = build_model(num_classes=4, dropout=0.3).to(device)
    checkpoint_path = "checkpoints/best_model_v6.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print("Modelo V6 carregado com sucesso.")
    
    target_layer = model.features[-1]
    
    imagem_teste = "dataset\\test\\Stricture\\1263_1263_image15753.png"
    
    caminho_procurado = os.path.normpath(imagem_teste)
    
    print(f"A procurar: {caminho_procurado}...")
    
    dataset_oficial = test_loader.dataset
    idx_encontrado = -1
    
    for i, item in enumerate(dataset_oficial.data):
        if os.path.normpath(item["image"]) == caminho_procurado:
            idx_encontrado = i
            break
            
    if idx_encontrado != -1:
        print(f"Imagem encontrada no índice {idx_encontrado} do Test Set!")
        
        dados_processados = dataset_oficial[idx_encontrado]
    
        input_tensor = dados_processados["image"].unsqueeze(0).to(device)
        label_idx = dados_processados["label"]
        true_class = CLASS_NAMES[label_idx]
        
        img_vis = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
        
        cam = GradCAM(model=model, target_layers=[target_layer])
        
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = torch.nn.functional.softmax(logits, dim=1)[0][pred_idx].item()
        
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        cam_image = show_cam_on_image(img_vis, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(11, 5), dpi=200)
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_vis)
        plt.title(f"Radiografia Original\nClasse Real: {true_class}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.title(f"Attention Map da Rede (V6)\nPred: {pred_class} ({confidence*100:.1f}%)")
        plt.axis('off')
        
        plt.tight_layout()
        save_path = "Stricture_1263_1263_image15753.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Mapa de atenção guardado com sucesso em: {save_path}")
        
    else:
        print(f"Erro: A imagem '{imagem_teste}' não existe no test set gerado pelo teu Dataloader.")