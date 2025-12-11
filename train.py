import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------------------
# Basic setup
# ---------------------------
torch.manual_seed(42)

DATA_DIR = "data"

# Standard ResNet transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),   # small augmentation boost
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Loading datasets
# ---------------------------
print("Loading datasets...")

train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_train)
val_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=transform_eval)
test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=transform_eval)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_data)}")
print(f"Val samples:   {len(val_data)}")
print(f"Test samples:  {len(test_data)}")

# ---------------------------
# Model Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pretrained ResNet18
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze the backbone to speed up the training 
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)   # faster learning since only FC trains

# ---------------------------
# Helper Function for Evaluation 
# ---------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0

# ---------------------------
# Training loop
# ---------------------------
EPOCHS = 3   # Fast training: adjust later once stable
print("\nStarting training...\n")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    val_acc = evaluate(model, val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "model/pneumonia_resnet18_fast.pth")
        print(" → Saved new best model!")

# ---------------------------
# Final test evaluation
# ---------------------------
print("\nRunning final test evaluation...")
test_acc = evaluate(model, test_loader)
print(f"Final Test Accuracy: {test_acc:.4f}")

print("\nTraining complete!")
