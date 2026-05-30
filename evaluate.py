import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from Dataloader import get_dataloaders, CLASS_NAMES
from Treino import build_model, evaluate_test
import wandb

CHECKPOINT = "checkpoints/best_model_v6.pth"
DATA_DIR   = "dataset"
BATCH_SIZE = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=0,
    )

    model = build_model(num_classes=4, dropout=0.3).to(device)

    wandb.init(
        project="cpre-miqr-classification",
        name="v6-standalone-eval", 
        config={"checkpoint": CHECKPOINT}
    )

    print("\nA iniciar avaliacao final...")
    evaluate_test(
        model=model, 
        test_loader=test_loader, 
        device=device, 
        class_names=class_names, 
        checkpoint_path=CHECKPOINT,
        label="V6_Final"
    )

    wandb.finish()

if __name__ == "__main__":
    main()