import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)
import matplotlib.pyplot as plt


# Custom wrapper to apply dynamic transforms to a dataset subset
class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Dataset Setup
root_dir = './data'

trainval_dataset = datasets.OxfordIIITPet(
    root=root_dir,
    split='trainval',
    target_types='category',
    download=True
)

test_dataset = datasets.OxfordIIITPet(
    root=root_dir,
    split='test',
    target_types='category',
    download=True
)

train_size = int(0.8 * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
generator = torch.Generator().manual_seed(42)

train_split, val_split = random_split(
    trainval_dataset,
    [train_size, val_size],
    generator=generator
)

# Extract class names for visualization
class_names = trainval_dataset.classes

# Model Initialization and Modification
num_classes = 37

cnn_weights = ResNet50_Weights.IMAGENET1K_V1
cnn = resnet50(weights=cnn_weights)
cnn_in_features = cnn.fc.in_features
cnn.fc = nn.Linear(in_features=cnn_in_features, out_features=num_classes)

vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
vit = vit_b_16(weights=vit_weights)
vit_in_features = vit.heads.head.in_features
vit.heads.head = nn.Linear(in_features=vit_in_features, out_features=num_classes)

# Input Transformations
cnn_default_transforms = cnn_weights.transforms()
vit_default_transforms = vit_weights.transforms()

# Compose training augmentations with default model transforms
cnn_train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    cnn_default_transforms
])

vit_train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    vit_default_transforms
])

# Apply transforms to splits using the wrapper
cnn_train_dataset = TransformedDataset(train_split, transform=cnn_train_transforms)
cnn_val_dataset = TransformedDataset(val_split, transform=cnn_default_transforms)
cnn_test_dataset = TransformedDataset(test_dataset, transform=cnn_default_transforms)

vit_train_dataset = TransformedDataset(train_split, transform=vit_train_transforms)
vit_val_dataset = TransformedDataset(val_split, transform=vit_default_transforms)
vit_test_dataset = TransformedDataset(test_dataset, transform=vit_default_transforms)

# DataLoaders
batch_size = 32

cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=batch_size, shuffle=True)
cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=batch_size, shuffle=False)
cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=batch_size, shuffle=False)

vit_train_loader = DataLoader(vit_train_dataset, batch_size=batch_size, shuffle=True)
vit_val_loader = DataLoader(vit_val_dataset, batch_size=batch_size, shuffle=False)
vit_test_loader = DataLoader(vit_test_dataset, batch_size=batch_size, shuffle=False)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# Training Function
def train_model(model, train_loader, val_loader, test_loader, model_name, optimizer, phase_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using ",device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    epochs = 10

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print(f"\nStarting {phase_name} for {model_name} on {device}...")

    total_time = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()

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
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time

        print(f"Epoch [{epoch + 1}/{epochs}] - Time: {epoch_time:.2f}s | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    avg_time_per_epoch = total_time / epochs
    print(f"Average time per epoch for {model_name} ({phase_name}): {avg_time_per_epoch:.2f}s")

    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = correct_test / total_test
    print(f"{model_name} Final Test Accuracy ({phase_name}): {test_acc:.4f}\n")

    return history, test_acc, avg_time_per_epoch


# Plotting the training results
def plot_history(history, model_name, phase_name):
    epochs_range = range(1, 11)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
    plt.title(f'{model_name} Loss per Epoch\n({phase_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy', marker='o')
    plt.title(f'{model_name} Accuracy per Epoch\n({phase_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_{phase_name.replace(" ", "_")}.png')
    plt.show()


# Utility to denormalize images for plotting
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


# Function to visualize correct and incorrect predictions
def visualize_predictions(model, test_loader, class_names, model_name, num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = preds[i].item()

                true_prob = probs[i, true_label].item()
                pred_prob = probs[i, pred_label].item()

                sample_info = (img, true_label, pred_label, true_prob, pred_prob)

                if true_label == pred_label:
                    if len(correct_samples) < num_samples:
                        correct_samples.append(sample_info)
                else:
                    if len(incorrect_samples) < num_samples:
                        incorrect_samples.append(sample_info)

                if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                    break
            if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                break

    # Plotting the samples
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 10))
    fig.suptitle(f"{model_name} Prediction Analysis", fontsize=16)

    # Plot correct predictions
    for idx, info in enumerate(correct_samples):
        img, t_lbl, p_lbl, t_prob, p_prob = info
        img = denormalize(img)
        img_np = img.permute(1, 2, 0).numpy()

        ax = axes[0, idx]
        ax.imshow(img_np)
        ax.axis('off')

        title = f"CORRECT\nTrue: {class_names[t_lbl]}\nPred: {class_names[p_lbl]}\nProb: {p_prob:.4f}"
        ax.set_title(title, color="green", fontsize=10)

    # Plot incorrect predictions
    for idx, info in enumerate(incorrect_samples):
        img, t_lbl, p_lbl, t_prob, p_prob = info
        img = denormalize(img)
        img_np = img.permute(1, 2, 0).numpy()

        ax = axes[1, idx]
        ax.imshow(img_np)
        ax.axis('off')

        title = f"INCORRECT\nTrue: {class_names[t_lbl]} (Prob: {t_prob:.4f})\nPred: {class_names[p_lbl]} (Prob: {p_prob:.4f})"
        ax.set_title(title, color="red", fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{model_name}_visualizations.png')
    plt.show()


# Execution Flow

# Phase 1: Linear Probing
print("Phase 1: Linear Probing")

for param in cnn.parameters():
    param.requires_grad = False
for param in cnn.fc.parameters():
    param.requires_grad = True

for param in vit.parameters():
    param.requires_grad = False
for param in vit.heads.head.parameters():
    param.requires_grad = True

cnn_optimizer_lp = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=1e-3)
cnn_history_lp, cnn_test_acc_lp, cnn_time_lp = train_model(cnn, cnn_train_loader, cnn_val_loader, cnn_test_loader,
                                                           "ResNet50",
                                                           cnn_optimizer_lp, "Linear_Probing")
plot_history(cnn_history_lp, "ResNet50", "Linear_Probing")

vit_optimizer_lp = torch.optim.Adam(filter(lambda p: p.requires_grad, vit.parameters()), lr=1e-3)
vit_history_lp, vit_test_acc_lp, vit_time_lp = train_model(vit, vit_train_loader, vit_val_loader, vit_test_loader,
                                                           "ViT-B_16",
                                                           vit_optimizer_lp, "Linear_Probing")
plot_history(vit_history_lp, "ViT-B_16", "Linear_Probing")

cnn_params = count_trainable_params(cnn)
vit_params = count_trainable_params(vit)

print(f"ResNet params: {cnn_params}")
print(f"ViT params: {vit_params}")
# Phase 2: Full Fine-Tuning

print("Phase 2: Full Fine-Tuning")

for param in cnn.parameters():
    param.requires_grad = True

for param in vit.parameters():
    param.requires_grad = True

cnn_ft_params = [
    {"params": [p for n, p in cnn.named_parameters() if n != 'fc.weight' and n != 'fc.bias'], "lr": 1e-5},
    {"params": cnn.fc.parameters(), "lr": 1e-3}
]
cnn_optimizer_ft = torch.optim.Adam(cnn_ft_params)

vit_ft_params = [
    {"params": [p for n, p in vit.named_parameters() if not n.startswith('heads.head')], "lr": 1e-5},
    {"params": vit.heads.head.parameters(), "lr": 1e-3}
]
vit_optimizer_ft = torch.optim.Adam(vit_ft_params)

cnn_history_ft, cnn_test_acc_ft, cnn_time_ft = train_model(cnn, cnn_train_loader, cnn_val_loader, cnn_test_loader,
                                                           "ResNet50",
                                                           cnn_optimizer_ft, "Full_Fine-Tuning")
plot_history(cnn_history_ft, "ResNet50", "Full_Fine-Tuning")

vit_history_ft, vit_test_acc_ft, vit_time_ft = train_model(vit, vit_train_loader, vit_val_loader, vit_test_loader,
                                                           "ViT-B_16",
                                                           vit_optimizer_ft, "Full_Fine-Tuning")
plot_history(vit_history_ft, "ViT-B_16", "Full_Fine-Tuning")

cnn_params = count_trainable_params(cnn)
vit_params = count_trainable_params(vit)

print(f"ResNet params: {cnn_params}")
print(f"ViT params: {vit_params}")
# Phase 3: Visualizations on the final Fine-Tuned models

print("Phase 3: Visualizations")

visualize_predictions(cnn, cnn_test_loader, class_names, "ResNet50")
visualize_predictions(vit, vit_test_loader, class_names, "ViT-B_16")

# Summary print for table filling
print("\n--- Summary for Table ---")
print(f"ResNet50 (Linear Probing): Test Acc: {cnn_test_acc_lp:.4f}, Avg Time/Epoch: {cnn_time_lp:.2f}s")
print(f"ResNet50 (Fine-Tuning): Test Acc: {cnn_test_acc_ft:.4f}, Avg Time/Epoch: {cnn_time_ft:.2f}s")
print(f"ViT-B/16 (Linear Probing): Test Acc: {vit_test_acc_lp:.4f}, Avg Time/Epoch: {vit_time_lp:.2f}s")
print(f"ViT-B/16 (Fine-Tuning): Test Acc: {vit_test_acc_ft:.4f}, Avg Time/Epoch: {vit_time_ft:.2f}s")

print("\nPhase 4: Extra Experiment - High LR on ViT (Testing Catastrophic Forgetting)")

# Re-initialize a fresh ViT model to start from scratch for this test
vit_failed_experiment = vit_b_16(weights=vit_weights)
vit_failed_in_features = vit_failed_experiment.heads.head.in_features
vit_failed_experiment.heads.head = nn.Linear(in_features=vit_failed_in_features, out_features=num_classes)

# Unfreeze all layers for full fine-tuning
for param in vit_failed_experiment.parameters():
    param.requires_grad = True

# HYPOTHESIS TEST: Using a unified, very high learning rate (1e-2) for the entire ViT backbone.
# We expect this to destroy the pre-trained weights and result in zero learning progress.
failed_optimizer = torch.optim.Adam(vit_failed_experiment.parameters(), lr=1e-2)


failed_vit_history, failed_vit_test_acc,failed_vit_time = train_model(
    vit_failed_experiment,
    vit_train_loader,
    vit_val_loader,
    vit_test_loader,
    "ViT-B_16_Failed",
    failed_optimizer,
    "High_LR_Failure",
)

plot_history(failed_vit_history, "ViT-B_16_Failed", "High_LR_Failure")
print(f"Failed ViT-B/16 (Fine-Tuning): Test Acc: {failed_vit_test_acc:.4f}, Avg Time/Epoch: {failed_vit_time:.2f}s")
visualize_predictions(vit_failed_experiment, vit_test_loader, class_names, "Failed ViT")
failed_vit_params = count_trainable_params(vit_failed_experiment)
print(f"Failed ViT params: {failed_vit_params}")

