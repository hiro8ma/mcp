#!/usr/bin/env python3
"""
Image Classifier MCP Server
手書き数字（MNIST）の画像分類を提供するMCPサーバー
"""

import base64
import io
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("Image Classifier")


# CNN Model Definition (Simple MNIST Classifier)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Global model instance
model = SimpleCNN()
model.eval()


def preprocess_image(image_data: str) -> torch.Tensor:
    """
    Base64エンコードされた画像データを前処理してテンソルに変換
    
    Args:
        image_data: Base64エンコードされた画像データ
    
    Returns:
        前処理済みのテンソル (1, 1, 28, 28)
    """
    # Base64デコード
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # グレースケール変換とリサイズ
    image = image.convert('L').resize((28, 28))
    
    # NumPy配列に変換し正規化
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # PyTorchテンソルに変換
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    # MNIST標準の正規化
    mean, std = 0.1307, 0.3081
    tensor = (tensor - mean) / std
    
    return tensor


@mcp.tool()
def classify_digit(image_base64: str) -> dict:
    """
    手書き数字画像（0-9）を分類します
    
    Base64エンコードされた画像データを受け取り、0-9の数字を予測します。
    画像は自動的に28x28ピクセルのグレースケールに変換されます。
    
    Args:
        image_base64: Base64エンコードされた画像データ（data:image/png;base64,...形式も可）
    
    Returns:
        予測結果を含む辞書:
        - predicted_digit: 予測された数字 (0-9)
        - confidence: 予測の信頼度 (0.0-1.0)
        - probabilities: 各数字の確率分布
    
    例:
        画像をBase64エンコードして渡すと、数字を予測します。
        手書きの数字、スキャンした数字、デジタルフォントなど対応。
    """
    try:
        # 画像の前処理
        tensor = preprocess_image(image_base64)
        
        # 予測
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][predicted].item()
        
        # 確率分布を辞書に変換
        prob_dict = {str(i): float(probabilities[0][i]) for i in range(10)}
        
        return {
            "predicted_digit": int(predicted.item()),
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "画像の処理中にエラーが発生しました"
        }


@mcp.tool()
def get_model_info() -> dict:
    """
    現在読み込まれているモデルの情報を取得します
    
    Returns:
        モデル情報を含む辞書:
        - model_type: モデルの種類
        - input_size: 入力画像サイズ
        - classes: 分類可能なクラス数
        - parameters: モデルのパラメータ数
    """
    param_count = sum(p.numel() for p in model.parameters())
    
    return {
        "model_type": "SimpleCNN",
        "architecture": "2 Conv layers + 2 FC layers",
        "input_size": "28x28 grayscale",
        "classes": 10,
        "class_names": [str(i) for i in range(10)],
        "total_parameters": param_count,
        "status": "Model loaded (untrained - for demo purposes)"
    }


if __name__ == "__main__":
    mcp.run()
