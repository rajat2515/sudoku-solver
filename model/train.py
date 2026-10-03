"""
PyTorch Training Script for Sudoku Digit Recognizer.
Supports training on standard MNIST (auto-downloaded) or custom image folders.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.network import DigitCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch Digit Classifier for Sudoku")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "custom"],
                        help="Dataset source: 'mnist' (auto-downloads) or 'custom' folder")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to dataset directory (used for custom or mnist storage)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--save-path", type=str, default="model/digit_cnn.pt",
                        help="Path to save the best model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on ('cuda' or 'cpu')")
    return parser.parse_args()


def get_data_loaders(dataset_type: str, data_dir: str, batch_size: int):
    # Transformation pipeline with data augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_type == "mnist":
        print(f"[INFO] Using torchvision MNIST dataset (stored in {data_dir})...")
        train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        val_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=val_transform)
        num_classes = 10
    else:
        print(f"[INFO] Loading custom image dataset from {data_dir}...")
        full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
        num_classes = len(full_dataset.classes)
        val_size = int(0.15 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_data, val_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, num_classes


def train():
    args = parse_args()
    device = torch.device(args.device)
    print(f"[INFO] Training device: {device}")

    # Create destination directory if needed
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    train_loader, val_loader, num_classes = get_data_loaders(args.dataset, args.data_dir, args.batch_size)
    print(f"[INFO] Total training batches: {len(train_loader)}, validation batches: {len(val_loader)}")
    print(f"[INFO] Number of classes: {num_classes}")

    model = DigitCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = 100.0 * correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / total_val
        val_epoch_acc = 100.0 * correct_val / total_val

        scheduler.step(val_epoch_loss)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  --> Saved new best checkpoint to {args.save_path} (Val Acc: {best_val_acc:.2f}%)")

    print(f"\n[INFO] Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[INFO] Weights saved to: {args.save_path}")


if __name__ == "__main__":
    train()
