# dataset/tiny_imagenet_loader.py
import os
import requests
from zipfile import ZipFile
from io import BytesIO
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def download_and_extract_dataset(url: str, extract_to: str = "data"):
    """Scarica e decomprime il dataset se non esiste già."""
    os.makedirs(extract_to, exist_ok=True)
    target_path = os.path.join(extract_to, "tiny-imagenet-200")
    if not os.path.exists(target_path):
        print("Downloading dataset...")
        response = requests.get(url)
        if response.status_code == 200:
            with ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall(extract_to)
            print("✅ Download and extraction complete!")
        else:
            raise Exception(f"Download failed with status {response.status_code}")
    else:
        print("✅ Dataset already exists.")
    return target_path

def get_data_loaders(dataset_root: str, batch_size: int = 64):
    """Crea i DataLoader per train e test."""
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_set = ImageFolder(root=os.path.join(dataset_root, "train"), transform=transform)
    test_set = ImageFolder(root=os.path.join(dataset_root, "val"), transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_set.classes)
