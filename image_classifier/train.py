#!/usr/bin/env python3
"""
SimpleCNN の MNIST 学習スクリプト
学習済み重みを mnist_cnn.pth に保存する
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from image_classifier_server import SimpleCNN


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # MNIST の標準正規化（サーバーの preprocess_image と同じ値）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(1, epochs + 1):
        # --- 学習 ---
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} [{batch_idx * len(data):>5d}/{len(train_loader.dataset)}]  Loss: {loss.item():.4f}")

        # --- テスト ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                correct += (output.argmax(dim=1) == target).sum().item()

        accuracy = correct / len(test_dataset) * 100
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.2f}%")

    # 重みを保存
    save_path = "mnist_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    train()
