import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, RepeatChanneld, NormalizeIntensityd

from Treino import build_model, CONFIG
from Dataloader import CLAHEd, CLASS_NAMES, IMG_SIZE

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_inference_transforms():
    return Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(IMG_SIZE, IMG_SIZE), mode="bilinear"),
        CLAHEd(keys=["image"], clip_limit=2.0),
        RepeatChanneld(keys=["image"], repeats=3),
        NormalizeIntensityd(keys=["image"], channel_wise=True)
    ])

def generate_cam(image_path, model, target_layer, device, save_path="cam_output.png"):
    transforms = get_inference_transforms()
    data = transforms({"image": image_path})
    
    input_tensor = torch.tensor(data["image"]).unsqueeze(0).to(device) # Shape: [1, 3, 512, 512]
    
    vis_transform = Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(IMG_SIZE, IMG_SIZE), mode="bilinear"),
        CLAHEd(keys=["image"], clip_limit=2.0),
        RepeatChanneld(keys=["image"], repeats=3)
    ])
    vis_data = vis_transform({"image": image_path})
    rgb_img = np.transpose(vis_data["image"], (1, 2, 0)) # Shape: [512, 512, 3]
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        pred_idx = logits.argmax(dim=1).item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = torch.nn.functional.softmax(logits, dim=1)[0][pred_idx].item()
    
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Radiografia Original (CLAHE)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"Atenção da Rede\nPred: {pred_class} ({confidence*100:.1f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f" Mapa de atenção guardado em: {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(num_classes=4, dropout=0.3).to(device)
    checkpoint_path = "checkpoints/best_model_v6.pth"
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print("Modelo carregado com sucesso!")
    
    target_layer = model.features[-1]
    
    imagem_teste = "dataset/test/Biliary_Leaks/1391_1391_image17608.png"
    
    if os.path.exists(imagem_teste):
        generate_cam(imagem_teste, model, target_layer, device, save_path="attention_map.png")
    else:
        print(f"Imagem de teste não encontrada em: {imagem_teste}")