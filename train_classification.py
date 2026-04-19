import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from data_loader import get_data_loaders
from classification import StrokeClassifier, StrokeTypeClassifier

def train_classification_model(model_choice: str, data_dir: str, epochs: int, batch_size: int, learning_rate: float, save_path: str):
    """
    Trains a ResNet-50 CNN model for either Stroke detection or Type classification.
    Requires data formatted as ImageFolder: dataset/train/Class1/, dataset/train/Class2/
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading dataset from: {data_dir} for task: {model_choice}")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    print(f"Detected classes: {class_names}")
    num_classes = len(class_names)
    
    # Save a small config mapping predicting classes for the app later
    import json
    with open(save_path + "_classes.json", "w") as f:
        json.dump(class_names, f)
    
    # 2. Initialize Model Architecture
    if model_choice.lower() == 'stroke':
        print("Initializing StrokeClassifier (ResNet-50 backbone)...")
        model = StrokeClassifier(num_classes=num_classes)
    elif model_choice.lower() == 'type':
        print("Initializing StrokeTypeClassifier (ResNet-50 backbone)...")
        model = StrokeTypeClassifier(num_classes=num_classes)
        
    model = model.to(device)
    
    # 3. Setup Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    
    # 4. Training Loop
    print("\n--- Starting Training Loop ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Phase
        print(f"Epoch {epoch+1}/{epochs}")
        train_bar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass (run data through CNN)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({'loss': loss.item()})
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🔥 New best validation accuracy! Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)
            
    print("\n--- Training Complete ---")
    
    # 5. Final Evaluation on unseen Test Set
    print("\n--- Running Inference on Test Set ---")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
         for inputs, labels in test_loader:
             inputs, labels = inputs.to(device), labels.to(device)
             outputs = model(inputs)
             _, predicted = torch.max(outputs, 1)
             test_total += labels.size(0)
             test_correct += (predicted == labels).sum().item()
             
    test_acc = 100 * test_correct / test_total
    print(f"🏆 Final Test Accuracy: {test_acc:.2f}%")
    print(f"Weights successfully saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-50 CNN Classifiers for Brain CT Scans.")
    parser.add_argument('--task', type=str, required=True, choices=['stroke', 'type'], help="Which classification model to train: 'stroke' (normal vs stroke) or 'type' (ischemic vs hemorrhagic)")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to your Kaggle dataset folder. MUST contain 'train', 'val', and 'test' subfolders containing class directories.")
    parser.add_argument('--epochs', type=int, default=15, help="Number of times to loop over the full dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Number of images per batch")
    parser.add_argument('--lr', type=float, default=0.0001, help="Adam optimizer learning rate")
    parser.add_argument('--save_path', type=str, default='best_model.pth', help="Where to save the .pth PyTorch weights file")
    
    args = parser.parse_args()
    
    train_classification_model(
        model_choice=args.task,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
