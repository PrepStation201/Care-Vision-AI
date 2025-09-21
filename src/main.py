import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.dataloader import MRI_Dataset
from models.train import ModelTrainer
from utils.model_loader import load_model

def main():
    # --- Configuration ---
    TRAIN_DIR = "path/to/your/training/data"  # <-- IMPORTANT: Update this path
    TEST_DIR = "path/to/your/testing/data"    # <-- IMPORTANT: Update this path
    EPOCHS = 15
    MODEL_NAME = "EfficientNet"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Data ---
    dataset = MRI_Dataset(TRAIN_DIR, TEST_DIR)
    train_loader, _ = dataset.get_loaders()

    # --- Load Model ---
    model = load_model(MODEL_NAME)
    model = model.to(DEVICE)

    # --- Training ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    trainer = ModelTrainer(model, train_loader, criterion, optimizer, scheduler, DEVICE)
    trainer.train(EPOCHS)

    # --- Save the Model ---
    torch.save(model.state_dict(), f"{MODEL_NAME}_mri_classifier.pth")
    print(f"Model saved as {MODEL_NAME}_mri_classifier.pth")

if __name__ == "__main__":
    main()