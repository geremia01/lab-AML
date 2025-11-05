import matplotlib.pyplot as plt
import numpy as np

def denormalize(image):
    image = image.to("cpu").numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def show_samples(dataloader, classes, num_classes=10):
    """Mostra alcune immagini di esempio."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    sampled = set()

    for inputs, labels in dataloader:
        for img, label in zip(inputs, labels):
            if len(sampled) == num_classes:
                break
            if label.item() not in sampled:
                sampled.add(label.item())
                ax = axes[len(sampled) - 1]
                ax.imshow(denormalize(img))
                ax.set_title(classes[label.item()], fontsize=8)
                ax.axis("off")
        if len(sampled) == num_classes:
            break
    plt.tight_layout()
    plt.show()

