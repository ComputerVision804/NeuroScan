import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
from collections import Counter

# ✅ Force CPU usage
device = torch.device("cpu")
print("⚠️ Training is set to run on CPU (GPU disabled due to performance issues).")

# ✅ Use smaller image size for faster training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ✅ Load datasets
train_dataset_full = datasets.ImageFolder("training", transform=transform)
test_dataset = datasets.ImageFolder("testing", transform=transform)

# ✅ Split training into training + validation (90%/10%)
train_size = int(0.9 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# ✅ Class mapping and sample count
print("Class-to-Index Mapping:", train_dataset_full.class_to_idx)
targets = [label for _, label in train_dataset_full.samples]
class_counts = Counter(targets)
print("Samples per class:")
for idx, count in class_counts.items():
    class_name = list(train_dataset_full.class_to_idx.keys())[list(train_dataset_full.class_to_idx.values()).index(idx)]
    print(f"  {class_name}: {count}")

# ✅ Compute class weights
total_samples = sum(class_counts.values())
weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

# ✅ Load smaller ViT model (tiny variant)
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=4)
model = model.to(device)

# ✅ Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Training loop with validation
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_acc = 100 * correct_train / total_train

    # ✅ Validation step
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_acc = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

# ✅ Save trained model
torch.save(model.state_dict(), "vit_tiny_brain_tumor_model.pth")
print("✅ Model saved as vit_tiny_brain_tumor_model.pth")
