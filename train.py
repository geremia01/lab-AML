from dataset.tiny_imagenet_loader import download_and_extract_dataset, get_data_loaders
from utils.visualization import show_samples

if __name__ == "__main__":
    dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_root = download_and_extract_dataset(dataset_url, extract_to="data")

    train_loader, test_loader, num_classes = get_data_loaders(dataset_root)
    print(f"✅ Dataset ready! {num_classes} classes.")

    # Mostra 10 immagini
    show_samples(train_loader, classes=train_loader.dataset.classes)
