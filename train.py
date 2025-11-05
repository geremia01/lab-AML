# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.tiny_imagenet_loader import download_and_extract_dataset, get_data_loaders
from utils.visualization import show_samples
import wandb

def main():
    # ----------------------
    # Step 3: Dataset Setup
    # ----------------------
    dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_root = download_and_extract_dataset(dataset_url, extract_to="data")

    train_loader, test_loader, num_classes = get_data_loaders(dataset_root)
    print(f"✅ Dataset ready! {num_classes} classes.")

    # Mostra 10 immagini di esempio
    show_samples(train_loader, classes=train_loader.dataset.classes)

    # ----------------------
    # Step 4: Initialize W&B
    # ----------------------
    wandb.init(project="tiny-imagenet-training", config={
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001
    })
    config = wandb.config

    # ----------------------
    # Step 4: Define Model
    # ----------------------
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 256), nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # ----------------------
    # Step 4: Training Loop
    # ----------------------
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        acc = 100.0 * correct / total

        # Log metrics su W&B
        wandb.log({"loss": avg_loss, "accuracy": acc})
        print(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    print("✅ Training completed!")

if __name__ == "__main__":
    main()
