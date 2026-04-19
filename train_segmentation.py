import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm
from segmentation_detection import UNet
from segmentation_dataset import SegmentationDataset

def train_segmentation_model(images_dir: str, masks_dir: str, epochs: int, batch_size: int, learning_rate: float, save_path: str):
    """
    Trains a complete U-Net CNN for predicting pixel-wise brain clot and lesion masks.
    Requires paired ground truth mask images (e.g., from Kaggle).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Images Path: {images_dir}")
    print(f"Masks Path: {masks_dir}")
    
    # 1. Load Custom Dataset & Split (80% Train, 20% Val)
    print("Initializing Segmentation Dataset...")
    full_dataset = SegmentationDataset(images_dir=images_dir, masks_dir=masks_dir, image_size=(256, 256))
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Total training pairs: {train_size} | Total validation pairs: {val_size}")
    
    # 2. Initialize CNN U-Net Architecture
    print("Initializing U-Net Architecture from scratch...")
    # Using 3 channels (RGB) input and 1 channel binary output (lesion vs background)
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # 3. Setup Loss Function and Optimizer
    # Since model forward() already contains torch.sigmoid at the very end in your file,
    # we use standard BCELoss (not BCEWithLogitsLoss).
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    # 4. Training Loop
    print("\n--- Starting Training Loop for U-Net ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        print(f"Epoch {epoch+1}/{epochs}")
        train_bar = tqdm(train_loader, desc="Training")
        
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through UNet
            preds = model(images)
            loss = criterion(preds, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix({'loss': loss.item()})
            
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
             for images, masks in val_loader:
                 images, masks = images.to(device), masks.to(device)
                 preds = model(images)
                 loss = criterion(preds, masks)
                 val_loss += loss.item() * images.size(0)
                 
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Train Loss (BCE): {train_loss:.4f} | Validation Loss (BCE): {val_loss:.4f}")
        
        # Save Best U-Net Weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"🔥 New best validation loss! Saving U-Net model to {save_path}")
            torch.save(model.state_dict(), save_path)
            
    print("\n--- U-Net Training Complete ---")
    print(f"Final trained weights available at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net CNN for Brain CT Segmentation.")
    parser.add_argument('--images_dir', type=str, required=True, help="Path to folder containing original brain CT images.")
    parser.add_argument('--masks_dir', type=str, required=True, help="Path to folder containing binary black/white masks.")
    parser.add_argument('--epochs', type=int, default=30, help="Number of times to loop over dataset.")
    parser.add_argument('--batch_size', type=int, default=8, help="Number of images per batch (U-Net is memory intensive).")
    parser.add_argument('--lr', type=float, default=0.0001, help="Adam optimizer learning rate.")
    parser.add_argument('--save_path', type=str, default='unet_weights.pth', help="Where to save the .pth U-Net PyTorch weights.")
    
    args = parser.parse_args()
    
    train_segmentation_model(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )