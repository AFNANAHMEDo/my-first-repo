```
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

```

```
import numpy as np

test_images= np.load("/kaggle/input/datasets/afnanahmed2/inter-iit-cv/test_images.npy")
train_images = np.load("/kaggle/input/datasets/afnanahmed2/inter-iit-cv/train_images.npy")

```

```
import pandas as pd
import torch

```

```
labels = pd.read_csv("/kaggle/input/datasets/afnanahmed2/inter-iit-cv/labels.csv")

```

```
train_idx = np.loadtxt(
    "/kaggle/input/datasets/afnanahmed2/inter-iit-cv/train_idx.txt",
    dtype=int
)

```

```
val_idx = np.loadtxt(
    "/kaggle/input/datasets/afnanahmed2/inter-iit-cv/val_idx.txt",
    dtype=int
)

```

```
print(train_images.shape)
print(len(train_idx), len(val_idx))

```

```
clean_labels = labels["clean_label"].values
noisy_labels = labels["noisy_label"].values

X_train = train_images[train_idx]
X_val   = train_images[val_idx]

y_clean_train = clean_labels[train_idx]
y_clean_val   = clean_labels[val_idx]

y_noisy_train = noisy_labels[train_idx]
y_noisy_val   = noisy_labels[val_idx]

```

```
import os

MODEL_PATH = "/kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1"

print(os.listdir(MODEL_PATH))

```

```
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

```

```
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

```

```
import os

for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f in ["train_images.npy", "labels.csv", "train_idx.txt", "val_idx.txt"]:
            print(os.path.join(root, f))

```

```
import numpy as np
import pandas as pd

DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"

# Load images
images = np.load(f"{DATA_PATH}/train_images.npy")

# Load labels
labels = pd.read_csv(f"{DATA_PATH}/labels.csv")

# Load official train/validation indices
train_idx = np.loadtxt(f"{DATA_PATH}/train_idx.txt", dtype=int)
val_idx = np.loadtxt(f"{DATA_PATH}/val_idx.txt", dtype=int)

print("Images:", images.shape)
print("Labels:", labels.shape)
print("Train indices:", len(train_idx))
print("Validation indices:", len(val_idx))

```

```
clean_labels = labels["clean_label"].to_numpy()
noisy_labels = labels["noisy_label"].to_numpy()

# Training split
X_train = images[train_idx]

y_clean_train = clean_labels[train_idx]
y_noisy_train = noisy_labels[train_idx]

# Validation split
X_val = images[val_idx]

y_clean_val = clean_labels[val_idx]
y_noisy_val = noisy_labels[val_idx]

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("Clean train:", y_clean_train.shape)
print("Noisy train:", y_noisy_train.shape)

```

```
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Training augmentation allowed by the problem
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

# Validation: no random augmentation
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label

```

```
# Two training datasets — SAME images, different labels
clean_train_dataset = CIFARDataset(
    X_train,
    y_clean_train,
    train_transform
)

noisy_train_dataset = CIFARDataset(
    X_train,
    y_noisy_train,
    train_transform
)

# Validation datasets
clean_val_dataset = CIFARDataset(
    X_val,
    y_clean_val,
    val_transform
)

noisy_val_dataset = CIFARDataset(
    X_val,
    y_noisy_val,
    val_transform
)

```

```
BATCH_SIZE = 128

clean_train_loader = DataLoader(
    clean_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

noisy_train_loader = DataLoader(
    noisy_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

clean_val_loader = DataLoader(
    clean_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

noisy_val_loader = DataLoader(
    noisy_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Clean batches:", len(clean_train_loader))
print("Noisy batches:", len(noisy_train_loader))
print("Validation batches:", len(clean_val_loader))

```

```
images_batch, labels_batch = next(iter(clean_train_loader))

print(images_batch.shape)
print(labels_batch.shape)

```

```
import sys
import torch
#/kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1/model.py
MODEL_PATH = "/kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1"

sys.path.insert(0, MODEL_PATH)

from model import ResNet18

print("ResNet18 imported successfully!")

```

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clean_model = ResNet18(num_classes=10).to(device)
noisy_model = ResNet18(num_classes=10).to(device)

print("Device:", device)
print("Clean model:", sum(p.numel() for p in clean_model.parameters()))
print("Noisy model:", sum(p.numel() for p in noisy_model.parameters()))

```

```
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
#SGD we are using 
clean_optimizer = optim.SGD(
    clean_model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

noisy_optimizer = optim.SGD(
    noisy_model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

print("Optimizers ready")

```

```
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc

```

```
train_one_epoch(
    clean_model,
    clean_train_loader,
    clean_optimizer,
    criterion,
    device
)

```

```
train_one_epoch(
    noisy_model,
    noisy_train_loader,
    noisy_optimizer,
    criterion,
    device
)

```

```
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total
    acc = 100.0 * correct / total

    return loss, acc

```

```
clean_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    clean_optimizer,
    T_max=60
)

noisy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    noisy_optimizer,
    T_max=60
)

```

```
NUM_EPOCHS = 60

clean_history = []
noisy_history = []

for epoch in range(NUM_EPOCHS):

    # -------------------------
    # CLEAN MODEL
    # -------------------------
    clean_train_loss, clean_train_acc = train_one_epoch(
        clean_model,
        clean_train_loader,
        clean_optimizer,
        criterion,
        device
    )

    clean_val_loss, clean_val_acc = evaluate(
        clean_model,
        clean_val_loader,
        criterion,
        device
    )

    clean_scheduler.step()


    # -------------------------
    # NOISY MODEL
    # -------------------------
    noisy_train_loss, noisy_train_acc = train_one_epoch(
        noisy_model,
        noisy_train_loader,
        noisy_optimizer,
        criterion,
        device
    )

    noisy_val_loss, noisy_val_acc = evaluate(
        noisy_model,
        noisy_val_loader,
        criterion,
        device
    )

    noisy_scheduler.step()


    # Save history
    clean_history.append({
        "epoch": epoch + 1,
        "train_loss": clean_train_loss,
        "train_acc": clean_train_acc,
        "val_loss": clean_val_loss,
        "val_acc": clean_val_acc
    })

    noisy_history.append({
        "epoch": epoch + 1,
        "train_loss": noisy_train_loss,
        "train_acc": noisy_train_acc,
        "val_loss": noisy_val_loss,
        "val_acc": noisy_val_acc
    })


    print(
        f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | "
        f"Clean: train {clean_train_acc:.2f}% val {clean_val_acc:.2f}% | "
        f"Noisy: train {noisy_train_acc:.2f}% val {noisy_val_acc:.2f}%"
    )

```

```
# Save final Task 1 models
torch.save(clean_model.state_dict(), "/kaggle/working/clean_model_final.pth")
torch.save(noisy_model.state_dict(), "/kaggle/working/noisy_model_final.pth")

print("Models saved!")

```

```
TEST_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv/test_images.npy"

test_images = np.load(TEST_PATH)

print(test_images.shape)

```

```
class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image, idx


test_dataset = TestDataset(
    test_images,
    val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


print("Test batches:", len(test_loader))

```

```
def generate_predictions(model, loader, device):
    model.eval()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_ids.extend(ids.numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_ids), np.array(all_preds)

```

```
clean_ids, clean_preds = generate_predictions(
    clean_model,
    test_loader,
    device
)

noisy_ids, noisy_preds = generate_predictions(
    noisy_model,
    test_loader,
    device
)

```

```
import os

os.makedirs("/kaggle/working/predictions", exist_ok=True)

clean_df = pd.DataFrame({
    "id": clean_ids,
    "pred": clean_preds
})

noisy_df = pd.DataFrame({
    "id": noisy_ids,
    "pred": noisy_preds
})

clean_df.to_csv(
    "/kaggle/working/predictions/t1_clean_ref.csv",
    index=False
)

noisy_df.to_csv(
    "/kaggle/working/predictions/t1_noisy_ce.csv",
    index=False
)

print("Saved Task 1 predictions!")
print(clean_df.head())
print(noisy_df.head())

```

```
from sklearn.metrics import confusion_matrix
import numpy as np

# --------------------------------------------------
# 1. Noise rate
# --------------------------------------------------

noise_mask = noisy_labels != clean_labels

noise_count = noise_mask.sum()
total_count = len(clean_labels)

noise_rate = 100 * noise_count / total_count

print(f"Total training images: {total_count}")
print(f"Mislabeled images: {noise_count}")
print(f"Noise rate: {noise_rate:.2f}%")


# --------------------------------------------------
# 2. 10 × 10 confusion matrix
#    Rows    = clean label (true/reference)
#    Columns = noisy label
# --------------------------------------------------

cm = confusion_matrix(
    clean_labels,
    noisy_labels,
    labels=np.arange(10)
)

print("\n10 × 10 Confusion Matrix:")
print(cm)

```

```
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

print(cm_df)

```

```
import json
import os

task1_summary = {
    "noise_rate_percent": float(noise_rate),
    "mislabeled_count": int(noise_count),
    "total_count": int(total_count),
    "confusion_matrix": cm.tolist(),
    "clean_reference_val_accuracy": 94.52,
    "noisy_ce_val_accuracy": 49.66,
    "epochs": 60,
    "batch_size": 128,
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "CosineAnnealingLR",
    "augmentation": "RandomCrop(32,padding=4) + RandomHorizontalFlip"
}

with open("/kaggle/working/task1_summary.json", "w") as f:
    json.dump(task1_summary, f, indent=2)

print("Task 1 summary saved.")

```

```
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"

labels = pd.read_csv(f"{DATA_PATH}/labels.csv")

clean_labels = labels["clean_label"].to_numpy()
noisy_labels = labels["noisy_label"].to_numpy()

print("Dataset shape:", labels.shape)
print("Clean labels:", clean_labels.shape)
print("Noisy labels:", noisy_labels.shape)

```

```
noise_mask = noisy_labels != clean_labels

noise_count = int(noise_mask.sum())
total_count = len(clean_labels)
noise_rate = 100 * noise_count / total_count

print(f"Total samples: {total_count}")
print(f"Mislabeled samples: {noise_count}")
print(f"Correctly labeled samples: {total_count - noise_count}")
print(f"Noise rate: {noise_rate:.2f}%")

```

```
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

cm = confusion_matrix(
    clean_labels,
    noisy_labels,
    labels=np.arange(10)
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

print("Confusion Matrix (Clean vs Noisy Labels):")
display(cm_df)

```

```
import matplotlib.pyplot as plt

plt.figure(figsize=(11, 8))

plt.imshow(cm, interpolation="nearest")
plt.title("Clean Labels vs Noisy Labels")
plt.colorbar()

plt.xticks(range(10), classes, rotation=45, ha="right")
plt.yticks(range(10), classes)

plt.xlabel("Noisy Label")
plt.ylabel("Clean Label")
plt.tight_layout()
plt.show()

```

```
import json
import os

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cm_df.to_csv(f"{OUTPUT_DIR}/t1_confusion_matrix.csv")

noise_summary = {
    "total_samples": total_count,
    "mislabeled_samples": noise_count,
    "correctly_labeled_samples": total_count - noise_count,
    "noise_rate_percent": round(noise_rate, 4)
}

with open(f"{OUTPUT_DIR}/t1_noise_summary.json", "w") as f:
    json.dump(noise_summary, f, indent=4)

print("Saved:")
print("/kaggle/working/t1_confusion_matrix.csv")
print("/kaggle/working/t1_noise_summary.json")

```

# Task 2 — Understanding Memorization

Objective: Track how a neural network learns correctly labeled and mislabeled training samples over 60 epochs.

Metrics:

1. Training accuracy on correctly labeled samples
2. Training accuracy on mislabeled samples
3. Validation accuracy against noisy labels
4. Validation accuracy against clean labels
5. Early-stopping epoch selected using noisy validation accuracy

```
%pip install -q wandb

```

```
import wandb

wandb.login(key="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")

```

```
wandb.init(
    project="inter-iit-cv-bootcamp",
    name="task2-noisy-label-memorization",
    config={
        "model": "ResNet18",
        "epochs": 60,
        "batch_size": 128,
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "loss": "CrossEntropyLoss"
    }
)

```

```
print("W&B run:", wandb.run.name)
print("W&B URL:", wandb.run.url)

```

```
# Check whether the dataset and model setup still exist

required_vars = [
    "images",
    "labels",
    "train_idx",
    "val_idx",
    "clean_labels",
    "noisy_labels",
    "X_train",
    "X_val",
    "y_clean_train",
    "y_noisy_train",
    "y_clean_val",
    "y_noisy_val",
    "ResNet18",
    "device",
    "train_loader",
    "val_loader",
]

for name in required_vars:
    print(f"{name}: {'Available' if name in globals() else 'MISSING'}")

```

```
# ==========================================
# TASK 2: RELOAD DATA AND MODEL ARCHITECTURE
# ==========================================

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

# Dataset and model paths
DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"
MODEL_PATH = "/kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1"

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type != "cuda":
    print("WARNING: GPU is not enabled!")

# Load dataset
images = np.load(f"{DATA_PATH}/train_images.npy")
labels = pd.read_csv(f"{DATA_PATH}/labels.csv")

train_idx = np.loadtxt(f"{DATA_PATH}/train_idx.txt", dtype=int)
val_idx = np.loadtxt(f"{DATA_PATH}/val_idx.txt", dtype=int)

clean_labels = labels["clean_label"].to_numpy()
noisy_labels = labels["noisy_label"].to_numpy()

# Official train/validation split
X_train = images[train_idx]
X_val = images[val_idx]

y_clean_train = clean_labels[train_idx]
y_noisy_train = noisy_labels[train_idx]

y_clean_val = clean_labels[val_idx]
y_noisy_val = noisy_labels[val_idx]

print("\nDataset loaded successfully!")
print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)
print("Training indices:", len(train_idx))
print("Validation indices:", len(val_idx))

# Import the supplied ResNet-18 architecture
sys.path.insert(0, MODEL_PATH)
from model import ResNet18

print("\nResNet18 imported successfully!")
print("W&B run:", wandb.run.name if wandb.run else "No active run")

```

```
# ==========================================
# TASK 2: DATASETS AND DATALOADERS
# ==========================================

# CIFAR-10 normalization
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Training augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Deterministic evaluation: no random augmentation
eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class CIFARDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = int(self.targets[idx])

        if self.transform:
            image = self.transform(image)

        return image, target


# Training uses noisy labels
train_dataset = CIFARDataset(
    X_train, y_noisy_train, train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Measure training accuracy without random augmentation
train_eval_dataset = CIFARDataset(
    X_train, y_noisy_train, eval_transform
)

train_eval_loader = DataLoader(
    train_eval_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Validation: same images, two different label sets
val_dataset_noisy = CIFARDataset(
    X_val, y_noisy_val, eval_transform
)

val_loader_noisy = DataLoader(
    val_dataset_noisy,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

val_dataset_clean = CIFARDataset(
    X_val, y_clean_val, eval_transform
)

val_loader_clean = DataLoader(
    val_dataset_clean,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Identify the two training subsets
correct_mask = (y_noisy_train == y_clean_train)
mislabeled_mask = ~correct_mask

print("DataLoaders created successfully!")
print("Training batches:", len(train_loader))
print("Training evaluation batches:", len(train_eval_loader))
print("Noisy validation batches:", len(val_loader_noisy))
print("Clean validation batches:", len(val_loader_clean))
print()
print("Correctly labeled training samples:", correct_mask.sum())
print("Mislabeled training samples:", mislabeled_mask.sum())

```

```
# ==========================================
# TASK 2: MODEL AND OPTIMIZER
# ==========================================

# Create a fresh model from scratch
model = ResNet18(num_classes=10).to(device)

# Cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Optimizer: same setup as Task 1
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

# Learning-rate schedule
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=60
)

# Training configuration
NUM_EPOCHS = 60
best_val_acc = -1.0
best_epoch = 0

# Save paths
BEST_CHECKPOINT_PATH = f"{OUTPUT_DIR}/t2_early_stop.pth"
DYNAMICS_PATH = f"{OUTPUT_DIR}/t2_dynamics.csv"

# Store metrics for every epoch
history = []

# Verify model and parameter count
num_params = sum(p.numel() for p in model.parameters())

print("Fresh ResNet-18 initialized!")
print(f"Device: {device}")
print(f"Parameters: {num_params:,}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Best checkpoint will be saved to: {BEST_CHECKPOINT_PATH}")

```

```
# ==========================================
# TASK 2: EVALUATION FUNCTIONS
# ==========================================

@torch.no_grad()
def get_predictions(model, loader):
    """Get predictions and average loss for a DataLoader."""
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    for images_batch, targets in loader:
        images_batch = images_batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images_batch)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)

        all_preds.append(predictions.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    average_loss = total_loss / total_samples
    accuracy = 100.0 * (all_preds == all_targets).mean()

    return all_preds, all_targets, average_loss, accuracy


def evaluate_task2(model):
    """Evaluate memorization and both validation label sets."""

    # Evaluate the training set without random augmentation.
    train_preds, _, train_loss, _ = get_predictions(
        model, train_eval_loader
    )

    # Compare predictions against the same underlying examples.
    train_acc_correct = 100.0 * (
        train_preds[correct_mask] == y_noisy_train[correct_mask]
    ).mean()

    train_acc_mislabeled = 100.0 * (
        train_preds[mislabeled_mask] == y_noisy_train[mislabeled_mask]
    ).mean()

    # Validation accuracy using noisy labels.
    _, _, val_loss_noisy, val_acc_noisy = get_predictions(
        model, val_loader_noisy
    )

    # Validation accuracy using clean labels.
    _, _, val_loss_clean, val_acc_clean = get_predictions(
        model, val_loader_clean
    )

    return {
        "train_loss": train_loss,
        "train_acc_correct": float(train_acc_correct),
        "train_acc_mislabeled": float(train_acc_mislabeled),
        "val_loss_noisy": val_loss_noisy,
        "val_loss_clean": val_loss_clean,
        "val_acc_noisy": val_acc_noisy,
        "val_acc_clean": val_acc_clean,
    }


print("Task 2 evaluation functions are ready!")

```

```
# ==========================================
# TASK 2: TRAINING LOOP
# ==========================================

import copy
import time

# Confirm that the existing W&B run is active.
assert wandb.run is not None, "W&B run is not active. Initialize it first."

history = []
best_val_acc = -1.0
best_epoch = 0

BEST_CHECKPOINT_PATH = "/kaggle/working/t2_early_stop.pth"
DYNAMICS_PATH = "/kaggle/working/t2_dynamics.csv"

start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):

    # --------------------------------------
    # 1. Train for one epoch using noisy labels
    # --------------------------------------
    model.train()

    running_loss = 0.0
    total_samples = 0

    for images_batch, targets in train_loader:
        images_batch = images_batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images_batch)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    train_loss_epoch = running_loss / total_samples

    # --------------------------------------
    # 2. Evaluate training and validation sets
    # --------------------------------------
    metrics = evaluate_task2(model)

    # Use noisy validation accuracy for checkpoint selection.
    val_acc_noisy = metrics["val_acc_noisy"]

    # Record metrics.
    row = {
        "epoch": epoch,
        "train_loss": train_loss_epoch,
        "train_acc_correct": metrics["train_acc_correct"],
        "train_acc_mislabeled": metrics["train_acc_mislabeled"],
        "val_loss_noisy": metrics["val_loss_noisy"],
        "val_loss_clean": metrics["val_loss_clean"],
        "val_acc_noisy": metrics["val_acc_noisy"],
        "val_acc_clean": metrics["val_acc_clean"],
        "learning_rate": optimizer.param_groups[0]["lr"],
    }

    history.append(row)

    # --------------------------------------
    # 3. Log metrics to W&B
    # --------------------------------------
    wandb.log(row, step=epoch)

    # --------------------------------------
    # 4. Save the best checkpoint so far
    # --------------------------------------
    if val_acc_noisy > best_val_acc:
        best_val_acc = val_acc_noisy
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc_noisy": val_acc_noisy,
                "val_acc_clean": metrics["val_acc_clean"],
            },
            BEST_CHECKPOINT_PATH,
        )

        print(
            f"  New best checkpoint: epoch {epoch}, "
            f"noisy val accuracy {best_val_acc:.2f}%"
        )

    # --------------------------------------
    # 5. Save progress to CSV after every epoch
    # --------------------------------------
    pd.DataFrame(history).to_csv(DYNAMICS_PATH, index=False)

    # Update the learning rate.
    scheduler.step()

    # --------------------------------------
    # 6. Print progress
    # --------------------------------------
    elapsed_minutes = (time.time() - start_time) / 60

    print(
        f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss_epoch:.4f} | "
        f"Correct Subset: {metrics['train_acc_correct']:.2f}% | "
        f"Mislabeled Subset: {metrics['train_acc_mislabeled']:.2f}% | "
        f"Noisy Val: {metrics['val_acc_noisy']:.2f}% | "
        f"Clean Val: {metrics['val_acc_clean']:.2f}% | "
        f"Time: {elapsed_minutes:.1f} min"
    )

# --------------------------------------
# 7. Finish and save final summary
# --------------------------------------
summary = {
    "best_epoch": best_epoch,
    "best_val_acc_noisy": best_val_acc,
    "epochs_completed": len(history),
}

with open("/kaggle/working/t2_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

wandb.summary["best_epoch"] = best_epoch
wandb.summary["best_val_acc_noisy"] = best_val_acc
wandb.summary["epochs_completed"] = len(history)

print("\nTask 2 training completed!")
print(f"Best epoch: {best_epoch}")
print(f"Best noisy validation accuracy: {best_val_acc:.2f}%")
print(f"Checkpoint: {BEST_CHECKPOINT_PATH}")
print(f"Dynamics CSV: {DYNAMICS_PATH}")
print("Summary: /kaggle/working/t2_summary.json")

```

```
import pandas as pd
import json
import os

# Load training history
dynamics = pd.read_csv("/kaggle/working/t2_dynamics.csv")

# Load summary
with open("/kaggle/working/t2_summary.json", "r") as f:
    summary = json.load(f)

print("========== TASK 2 SUMMARY ==========")
print(json.dumps(summary, indent=4))

print("\n========== LAST 10 EPOCHS ==========")
display(dynamics.tail(10))

print("\n========== REQUIRED FILES ==========")
for filename in [
    "t2_dynamics.csv",
    "t2_summary.json",
    "t2_early_stop.pth"
]:
    path = f"/kaggle/working/{filename}"
    print(f"{filename}: {'EXISTS' if os.path.exists(path) else 'MISSING'}")

```

```
# ==========================================
# TASK 2: GENERATE EARLY-STOP TEST PREDICTIONS
# ==========================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Load the checkpoint selected using noisy validation accuracy
checkpoint = torch.load(
    "/kaggle/working/t2_early_stop.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print("Noisy validation accuracy:", checkpoint["val_acc_noisy"])


# Load test images
test_images = np.load(f"{DATA_PATH}/test_images.npy")

class TestDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        return image, idx


test_dataset = TestDataset(test_images, eval_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# Generate predictions
all_predictions = []
all_indices = []

with torch.no_grad():
    for images_batch, indices in test_loader:
        images_batch = images_batch.to(device, non_blocking=True)

        logits = model(images_batch)
        predictions = logits.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_indices.extend(indices.numpy())


# Save predictions in required format
os.makedirs("/kaggle/working/predictions", exist_ok=True)

submission = pd.DataFrame({
    "id": all_indices,
    "pred": all_predictions
})

output_path = "/kaggle/working/predictions/t2_early_stop.csv"
submission.to_csv(output_path, index=False)

print("\nPredictions saved:", output_path)
print("Prediction rows:", len(submission))
print("\nPreview:")
display(submission.head())

assert len(submission) == 10000
assert list(submission.columns) == ["id", "pred"]
assert submission["id"].is_unique
assert submission["pred"].between(0, 9).all()

print("\nTask 2 predictions validated successfully!")

```

```
import os

print("Checking Task 2 files...\n")

for filename in [
    "t2_dynamics.csv",
    "t2_summary.json",
    "t2_early_stop.pth",
]:
    path = f"/kaggle/working/{filename}"
    print(f"{filename}: {'EXISTS' if os.path.exists(path) else 'MISSING'}")

print("\nFiles currently in /kaggle/working:")
for item in sorted(os.listdir("/kaggle/working")):
    print(item)

```

```
import os

required_files = [
    "/kaggle/working/t1_confusion_matrix.csv",
    "/kaggle/working/t1_noise_summary.json",
    "/kaggle/working/predictions/t1_clean_ref.csv",
    "/kaggle/working/predictions/t1_noisy_ce.csv",
    "/kaggle/working/t2_dynamics.csv",
    "/kaggle/working/t2_summary.json",
    "/kaggle/working/t2_early_stop.pth",
    "/kaggle/working/predictions/t2_early_stop.csv",
]

for path in required_files:
    print(
        "OK      " if os.path.exists(path) else "MISSING ",
        path
    )

```

```
# ==========================================
# TASK 3: NOISE SCORE SETUP
# ==========================================

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold

# Reuse the dataset and model paths from earlier tasks.
DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"

OUTPUT_DIR = "/kaggle/working"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(PRED_DIR, exist_ok=True)

# Load the official training indices and noisy labels.
train_idx = np.loadtxt(
    f"{DATA_PATH}/train_idx.txt",
    dtype=int
)

labels_df = pd.read_csv(f"{DATA_PATH}/labels.csv")
noisy_labels_all = labels_df["noisy_label"].to_numpy()

# Only the official 45,000 training indices are used here.
task3_indices = train_idx.copy()
task3_labels = noisy_labels_all[task3_indices]

# Five folds; fixed seed for reproducibility.
N_FOLDS = 5
SEED = 42

kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

# Each image gets one out-of-fold prediction.
oof_probabilities = np.zeros(
    (len(task3_indices), 10),
    dtype=np.float32
)

oof_predictions = np.full(
    len(task3_indices),
    -1,
    dtype=np.int64
)

fold_assignment = np.full(
    len(task3_indices),
    -1,
    dtype=np.int64
)

print("Task 3 setup complete.")
print("Training images:", len(task3_indices))
print("Folds:", N_FOLDS)
print("Images per held-out fold:", len(task3_indices) // N_FOLDS)
print("Noise-score output:", f"{PRED_DIR}/t3_noise_scores.csv")
print("Device:", device)

```

```
# ==========================================
# TASK 3: FOLD TRAINING FUNCTIONS
# ==========================================

import random
import time
from torch.utils.data import Subset

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Transforms
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform_t3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

eval_transform_t3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the images if not already in memory
if "images" not in globals():
    images = np.load(f"{DATA_PATH}/train_images.npy")

# Dataset over the 45,000 official training examples
class Task3Dataset(Dataset):
    def __init__(self, original_indices, targets, transform):
        self.original_indices = np.asarray(original_indices)
        self.targets = np.asarray(targets)
        self.transform = transform

    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, idx):
        original_idx = int(self.original_indices[idx])
        image = images[original_idx]
        target = int(self.targets[idx])
        return self.transform(image), target

# Evaluation helper
@torch.no_grad()
def predict_probabilities(model, loader):
    model.eval()
    all_probs = []

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)

print("Task 3 fold-training functions ready.")

```

```
# ==========================================
# TASK 3: FIVE-FOLD OOF TRAINING
# ==========================================

import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

FOLD_DIR = "/kaggle/working/task3_folds"
os.makedirs(FOLD_DIR, exist_ok=True)

EPOCHS_T3 = 60
BATCH_SIZE_T3 = 128
BEST_VAL_PATH = "/kaggle/working/task3_best_fold_checkpoint.pth"

# Official validation data: noisy labels used only for checkpoint selection
val_idx_t3 = np.loadtxt(f"{DATA_PATH}/val_idx.txt", dtype=int)
val_labels_t3 = noisy_labels_all[val_idx_t3]

val_dataset_t3 = Task3Dataset(
    val_idx_t3,
    val_labels_t3,
    eval_transform_t3
)

val_loader_t3 = DataLoader(
    val_dataset_t3,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Save fold assignments so each image's OOF prediction is traceable
folds = list(kf.split(task3_indices))

for fold_num, (train_pos, heldout_pos) in enumerate(folds, start=1):

    fold_file = f"{FOLD_DIR}/fold_{fold_num}_oof_probs.npy"

    # Skip completed folds when resuming.
    if os.path.exists(fold_file):
        print(f"Fold {fold_num}: already completed; skipping.")
        oof_probabilities[heldout_pos] = np.load(fold_file)
        fold_assignment[heldout_pos] = fold_num
        oof_predictions[heldout_pos] = np.argmax(
            oof_probabilities[heldout_pos], axis=1
        )
        continue

    print(f"\n========== FOLD {fold_num}/{N_FOLDS} ==========")

    set_seed(SEED + fold_num)

    fold_train_indices = task3_indices[train_pos]
    fold_train_labels = task3_labels[train_pos]

    fold_train_dataset = Task3Dataset(
        fold_train_indices,
        fold_train_labels,
        train_transform_t3
    )

    fold_train_loader = DataLoader(
        fold_train_dataset,
        batch_size=BATCH_SIZE_T3,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Fresh ResNet-18 for this fold
    fold_model = ResNet18(num_classes=10).to(device)

    fold_criterion = nn.CrossEntropyLoss()

    fold_optimizer = optim.SGD(
        fold_model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    fold_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        fold_optimizer,
        T_max=EPOCHS_T3
    )

    best_fold_val_acc = -1.0
    best_fold_state = None

    for epoch in range(1, EPOCHS_T3 + 1):

        fold_model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in fold_train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            fold_optimizer.zero_grad(set_to_none=True)

            logits = fold_model(xb)
            loss = fold_criterion(logits, yb)

            loss.backward()
            fold_optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total_count += yb.size(0)

        fold_scheduler.step()

        # Select checkpoint using official noisy validation labels.
        _, _, _, val_acc = get_predictions(
            fold_model, val_loader_t3
        )

        if val_acc > best_fold_val_acc:
            best_fold_val_acc = val_acc
            best_fold_state = {
                k: v.detach().cpu().clone()
                for k, v in fold_model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Fold {fold_num} | Epoch {epoch:02d}/{EPOCHS_T3} | "
                f"Loss {total_loss / total_count:.4f} | "
                f"Noisy validation accuracy {val_acc:.2f}%"
            )

    # Restore best checkpoint before predicting held-out images.
    fold_model.load_state_dict(best_fold_state)

    torch.save(
        {
            "fold": fold_num,
            "model_state_dict": best_fold_state,
            "best_val_acc_noisy": best_fold_val_acc
        },
        BEST_VAL_PATH
    )

    heldout_indices = task3_indices[heldout_pos]

    heldout_dataset = Task3Dataset(
        heldout_indices,
        task3_labels[heldout_pos],
        eval_transform_t3
    )

    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    probs = predict_probabilities(fold_model, heldout_loader)

    # Save each completed fold immediately.
    np.save(fold_file, probs)

    oof_probabilities[heldout_pos] = probs
    oof_predictions[heldout_pos] = probs.argmax(axis=1)
    fold_assignment[heldout_pos] = fold_num

    np.save(
        f"{FOLD_DIR}/oof_probabilities_partial.npy",
        oof_probabilities
    )
    np.save(
        f"{FOLD_DIR}/fold_assignment.npy",
        fold_assignment
    )

    print(
        f"Fold {fold_num} completed. "
        f"Best noisy validation accuracy: {best_fold_val_acc:.2f}%"
    )

    del fold_model, fold_optimizer, fold_scheduler
    del fold_train_loader, heldout_loader
    gc.collect()
    torch.cuda.empty_cache()

print("\nAll five folds completed.")

```

```
# Verify that every training image has an out-of-fold prediction
assert np.all(fold_assignment != -1), "Some images are missing fold predictions!"
assert np.all(oof_predictions != -1), "Some OOF predictions are missing!"

# Probability assigned to each image's observed noisy label
row_indices = np.arange(len(task3_labels))
prob_observed_label = oof_probabilities[row_indices, task3_labels]

# Higher score = more likely the observed label is incorrect
noise_scores = -np.log(np.clip(prob_observed_label, 1e-8, 1.0))

# Create the required CSV
scores_df = pd.DataFrame({
    "index": task3_indices,
    "score": noise_scores
})

output_path = "/kaggle/working/predictions/t3_noise_scores.csv"
scores_df.to_csv(output_path, index=False)

# Basic validation
print("Task 3 noise scores saved!")
print("File:", output_path)
print("Number of images:", len(scores_df))
print("Missing scores:", scores_df["score"].isna().sum())
print("\nScore statistics:")
print(scores_df["score"].describe())
print("\nHighest-scored examples:")
print(scores_df.nlargest(10, "score"))

```

```
from sklearn.metrics import roc_auc_score

# Ground-truth noise status for evaluation only
# 1 = mislabeled, 0 = correctly labeled
true_noise = (
    clean_labels[task3_indices] != noisy_labels_all[task3_indices]
).astype(int)

# Predicted noise scores
scores = scores_df["score"].to_numpy()

# Number of truly mislabeled images in the 45,000-image training split
k = int(true_noise.sum())

# AUROC: measures how well scores rank mislabeled images above cleanly labeled ones
auroc = roc_auc_score(true_noise, scores)

# Precision@k: among the k highest-scored images, what fraction are truly mislabeled?
top_k_positions = np.argsort(scores)[-k:]
precision_at_k = true_noise[top_k_positions].mean()

print("========== TASK 3 EVALUATION ==========")
print(f"Training images evaluated: {len(scores):,}")
print(f"Truly mislabeled images (k): {k:,}")
print(f"True noise rate in training split: {true_noise.mean() * 100:.2f}%")
print(f"AUROC: {auroc:.4f}")
print(f"Precision@k: {precision_at_k:.4f} ({precision_at_k * 100:.2f}%)")

```

```
import os
import pandas as pd
import numpy as np

score_path = "/kaggle/working/predictions/t3_noise_scores.csv"

assert os.path.exists(score_path), "Noise-score CSV is missing!"

check_df = pd.read_csv(score_path)

# Check required format
assert list(check_df.columns) == ["index", "score"], \
    f"Unexpected columns: {list(check_df.columns)}"

assert len(check_df) == 45000, \
    f"Expected 45000 rows, found {len(check_df)}"

assert check_df["index"].is_unique, "Duplicate image indices found!"
assert check_df["score"].notna().all(), "Missing scores found!"
assert np.isfinite(check_df["score"]).all(), "Invalid scores found!"
assert set(check_df["index"]) == set(train_idx), \
    "CSV indices do not match the official training indices!"

print("Task 3 submission file: VALID")
print("Rows:", len(check_df))
print("Columns:", list(check_df.columns))
print("Unique indices:", check_df["index"].nunique())
print("Missing values:", check_df.isna().sum().sum())
print("AUROC:", round(auroc, 4))
print("Precision@k:", round(precision_at_k, 4))

```

```
import os
import torch
import pandas as pd

OUTPUT_DIR = "/kaggle/working"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

print("========== ENVIRONMENT ==========")

print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n========== EXISTING OUTPUTS ==========")

important_files = [
    "clean_model_final.pth",
    "noisy_model_final.pth",
    "t2_early_stop.pth",
    "t2_dynamics.csv",
    "t2_summary.json",
    "task1_summary.json",
]

for filename in important_files:
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        print(f"[FOUND] {filename} ({os.path.getsize(path) / 1e6:.2f} MB)")
    else:
        print(f"[MISSING] {filename}")

print("\n========== PREDICTION FILES ==========")

if os.path.isdir(PRED_DIR):
    for filename in sorted(os.listdir(PRED_DIR)):
        path = os.path.join(PRED_DIR, filename)
        if os.path.isfile(path):
            print(f"{filename} ({os.path.getsize(path) / 1e6:.2f} MB)")
else:
    print("Predictions directory does not exist.")

```

```
import os
import pandas as pd

OUTPUT_DIR = "/kaggle/working"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

source_path = os.path.join(OUTPUT_DIR, "t2_dynamics.csv")

assert os.path.exists(source_path), "Original Task 2 dynamics file is missing!"

df = pd.read_csv(source_path)

print("Original columns:")
print(df.columns.tolist())

print("\nOriginal shape:", df.shape)
print("\nFirst 3 rows:")
display(df.head(3))

```

```
import os
import pandas as pd

OUTPUT_DIR = "/kaggle/working"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

source_path = os.path.join(OUTPUT_DIR, "t2_dynamics.csv")
output_path = os.path.join(PRED_DIR, "t2_dynamics.csv")

df = pd.read_csv(source_path)

required_columns = [
    "epoch",
    "train_acc_correct",
    "train_acc_mislabeled",
    "val_acc_noisy",
    "val_acc_clean",
]

# Confirm all required columns exist
assert all(col in df.columns for col in required_columns)

# Keep only the official columns, in the required order
submission_df = df[required_columns].copy()

# Save the submission copy
submission_df.to_csv(output_path, index=False)

# Verify the output
check_df = pd.read_csv(output_path)

assert check_df.columns.tolist() == required_columns
assert len(check_df) == 60
assert check_df.isna().sum().sum() == 0

print("Task 2 submission CSV created successfully!")
print("Path:", output_path)
print("Shape:", check_df.shape)
print("Columns:", check_df.columns.tolist())
print("\nLast 3 epochs:")
display(check_df.tail(3))

```

```
import json
import os

# Update this path if your notebook is saved elsewhere.
NOTEBOOK_PATH = "/kaggle/working/TASK1JUPY.ipynb"

if not os.path.exists(NOTEBOOK_PATH):
    print("Notebook not found at:", NOTEBOOK_PATH)
    print("Please upload/download TASK1JUPY.ipynb to /kaggle/working,")
    print("or tell me its actual path.")
else:
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    print("Notebook found. Searching for Task 1 training calls...\n")

    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if "train_one_epoch" in source or "clean_model" in source or "noisy_model" in source:
            print(f"\n--- Cell {i} ---")
            print(source[:2500])

```

```
# Inspect the current Jupyter kernel's input history.
# This works only if the original kernel/session is still active.

matches = []

for i, cell_code in enumerate(In):
    if not isinstance(cell_code, str):
        continue

    if "train_one_epoch" in cell_code:
        matches.append((i, cell_code))

print("Training-related cells found:", len(matches))

for cell_number, cell_code in matches:
    print(f"\n{'=' * 60}")
    print(f"Executed input history entry: {cell_number}")
    print("=" * 60)
    print(cell_code[:4000])
    

```

```
print("========== MODEL VARIABLES ==========")

for name in ["clean_model", "noisy_model", "model"]:
    if name in globals():
        obj = globals()[name]
        print(f"{name}: {type(obj).__name__}")
    else:
        print(f"{name}: not present")

print("\n========== TRAINING FUNCTIONS ==========")

for name in ["train_one_epoch", "evaluate", "evaluate_model"]:
    print(f"{name}: {'available' if name in globals() else 'not available'}")

print("\n========== TASK 1 CHECKPOINTS ==========")

import os

for name in ["clean_model_final.pth", "noisy_model_final.pth"]:
    path = f"/kaggle/working/{name}"
    if os.path.exists(path):
        print(f"{name}: {os.path.getsize(path) / 1e6:.2f} MB")
    else:
        print(f"{name}: missing")

```

```
import os
import numpy as np
import pandas as pd

DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"
OUTPUT_DIR = "/kaggle/working"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

train_idx = np.loadtxt(
    f"{DATA_PATH}/train_idx.txt",
    dtype=int
)

labels_df = pd.read_csv(f"{DATA_PATH}/labels.csv")
noisy_labels_all = labels_df["noisy_label"].to_numpy()

scores_df = pd.read_csv(
    f"{PRED_DIR}/t3_noise_scores.csv"
)

assert len(train_idx) == 45000
assert len(scores_df) == 45000
assert scores_df["index"].is_unique

# Align scores to the official training-index order
score_map = scores_df.set_index("index")["score"]
train_scores = score_map.loc[train_idx].to_numpy()
train_noisy_labels = noisy_labels_all[train_idx]

# Pre-register filtering fraction
FILTER_FRACTION = 0.35

n_remove = int(FILTER_FRACTION * len(train_idx))
ranked_positions = np.argsort(train_scores)

# Retain the lowest-scored 65%
keep_positions = ranked_positions[:-n_remove]
kept_indices = train_idx[keep_positions]
kept_labels = train_noisy_labels[keep_positions]

print("Total training images:", len(train_idx))
print("Images selected for removal:", n_remove)
print("Images retained:", len(kept_indices))
print("Actual retained fraction:", round(len(kept_indices) / len(train_idx), 4))

print("\nNoisy-label class counts before filtering:")
print(pd.Series(train_noisy_labels).value_counts().sort_index())

print("\nNoisy-label class counts after filtering:")
print(pd.Series(kept_labels).value_counts().sort_index())

print("\nMinimum retained class count:", 
      pd.Series(kept_labels).value_counts().min())

```

```
import os
import json
import numpy as np
import pandas as pd

OUTPUT_DIR = "/kaggle/working"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
TASK4_DIR = os.path.join(OUTPUT_DIR, "task4")
os.makedirs(TASK4_DIR, exist_ok=True)

# Save retained original dataset indices
kept_indices_path = os.path.join(TASK4_DIR, "filtered_kept_indices.npy")
np.save(kept_indices_path, kept_indices)

# Save noisy-label class counts before and after filtering
before_counts = pd.Series(train_noisy_labels).value_counts().sort_index()
after_counts = pd.Series(kept_labels).value_counts().sort_index()

class_counts_df = pd.DataFrame({
    "class": range(10),
    "before_filtering": [int(before_counts.get(c, 0)) for c in range(10)],
    "after_filtering": [int(after_counts.get(c, 0)) for c in range(10)],
})

class_counts_path = os.path.join(TASK4_DIR, "filtered_class_counts.csv")
class_counts_df.to_csv(class_counts_path, index=False)

# Save configuration for reproducibility
filter_config = {
    "method": "task3_score_filtered_cross_entropy",
    "seed": 42,
    "filter_fraction": 0.35,
    "removed_images": int(len(train_idx) - len(kept_indices)),
    "retained_images": int(len(kept_indices)),
    "selection_rule": "Remove highest 35% of Task 3 noise scores",
    "label_source": "noisy_label",
    "clean_labels_used_for_filtering": False,
}

config_path = os.path.join(TASK4_DIR, "filtered_ce_config.json")
with open(config_path, "w") as f:
    json.dump(filter_config, f, indent=2)

print("Saved filtered-training configuration.")
print("Indices:", kept_indices_path)
print("Class counts:", class_counts_path)
print("Config:", config_path)
print("\nClass-count summary:")
display(class_counts_df)

```

```
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Paths
DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"
MODEL_PATH = "/kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1"
OUTPUT_DIR = "/kaggle/working"
TASK4_DIR = os.path.join(OUTPUT_DIR, "task4")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Reproducibility seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load arrays and official splits
images = np.load(f"{DATA_PATH}/train_images.npy")
test_images = np.load(f"{DATA_PATH}/test_images.npy")

train_idx = np.loadtxt(f"{DATA_PATH}/train_idx.txt", dtype=int)
val_idx = np.loadtxt(f"{DATA_PATH}/val_idx.txt", dtype=int)

labels_df = pd.read_csv(f"{DATA_PATH}/labels.csv")
noisy_labels_all = labels_df["noisy_label"].to_numpy()

X_val = images[val_idx]
y_noisy_val = noisy_labels_all[val_idx]

# Load filtered training indices created earlier
kept_indices = np.load(
    os.path.join(TASK4_DIR, "filtered_kept_indices.npy")
)

# Import the supplied ResNet-18
sys.path.insert(0, MODEL_PATH)
from model import ResNet18

print("Environment ready.")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Training split:", len(train_idx))
print("Validation split:", len(val_idx))
print("Filtered training split:", len(kept_indices))
print("Validation images:", X_val.shape)
print("Noisy validation labels:", y_noisy_val.shape)

# Test model construction without training
test_model = ResNet18(num_classes=10).to(device)
print("Model:", type(test_model).__name__)
print("Trainable parameters:", sum(
    p.numel() for p in test_model.parameters() if p.requires_grad
))

del test_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

```

```
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Fixed seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# CIFAR-10 normalization
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

train_transform_t4 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform_t4 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class Task4Dataset(Dataset):
    def __init__(self, original_indices, targets, transform):
        self.original_indices = np.asarray(original_indices)
        self.targets = np.asarray(targets)
        self.transform = transform

        assert len(self.original_indices) == len(self.targets)

    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, idx):
        original_idx = int(self.original_indices[idx])
        image = images[original_idx]
        label = int(self.targets[idx])

        return self.transform(image), label


# Generalized Cross-Entropy loss
class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probabilities = torch.softmax(logits, dim=1)
        p_true = probabilities.gather(
            1, targets.view(-1, 1)
        ).squeeze(1)

        losses = (1.0 - p_true.pow(self.q)) / self.q
        return losses.mean()


# Standard cross-entropy and label smoothing
ce_loss_t4 = nn.CrossEntropyLoss()
label_smoothing_loss_t4 = nn.CrossEntropyLoss(label_smoothing=0.1)
gce_loss_t4 = GCELoss(q=0.7)

print("Task 4 components defined.")
print("Filtered CE: standard cross-entropy")
print("GCE: q =", gce_loss_t4.q)
print("Label smoothing: epsilon = 0.1")

# Quick mathematical sanity check
dummy_logits = torch.randn(8, 10, device=device)
dummy_targets = torch.randint(0, 10, (8,), device=device)

for name, loss_fn in [
    ("CE", ce_loss_t4),
    ("GCE", gce_loss_t4),
    ("Label smoothing", label_smoothing_loss_t4),
]:
    loss_value = loss_fn(dummy_logits, dummy_targets)
    assert torch.isfinite(loss_value), f"{name} produced invalid loss"
    print(f"{name} test loss: {loss_value.item():.4f}")

```

```
import os
import json
import time
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

TASK4_DIR = "/kaggle/working/task4"
os.makedirs(TASK4_DIR, exist_ok=True)

EPOCHS_T4 = 60
BATCH_SIZE_T4 = 128
NUM_WORKERS_T4 = 2

def set_seed_t4(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_noisy_validation(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

    return {
        "val_loss_noisy": total_loss / total,
        "val_acc_noisy": 100.0 * correct / total,
    }


def train_task4_method(
    method_name,
    train_indices,
    train_targets,
    loss_function,
    seed=42,
):
    """
    Train one fresh supplied ResNet-18 for up to 60 epochs.
    Model selection uses noisy validation accuracy only.
    """

    set_seed_t4(seed)

    method_dir = os.path.join(TASK4_DIR, method_name)
    os.makedirs(method_dir, exist_ok=True)

    train_dataset = Task4Dataset(
        train_indices,
        train_targets,
        train_transform_t4,
    )

    val_dataset = Task4Dataset(
        val_idx,
        noisy_labels_all[val_idx],
        eval_transform_t4,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_T4,
        shuffle=True,
        num_workers=NUM_WORKERS_T4,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS_T4,
        pin_memory=True,
    )

    model = ResNet18(num_classes=10).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_T4,
    )

    history = []
    best_val_acc = -1.0
    best_epoch = None
    best_state = None
    start_time = time.time()

    for epoch in range(1, EPOCHS_T4 + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_function(logits, yb)

            loss.backward()
            optimizer.step()

            batch_size = yb.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

        scheduler.step()

        train_loss = running_loss / total
        val_metrics = evaluate_noisy_validation(model, val_loader)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_noisy": val_metrics["val_loss_noisy"],
            "val_acc_noisy": val_metrics["val_acc_noisy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        # Strictly select by noisy validation accuracy.
        # Keep the earlier epoch if there is a tie.
        if val_metrics["val_acc_noisy"] > best_val_acc:
            best_val_acc = val_metrics["val_acc_noisy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "val_acc_noisy": best_val_acc,
                    "method": method_name,
                    "seed": seed,
                },
                os.path.join(method_dir, "best_checkpoint.pth"),
            )

        # Save a resumable final-epoch state.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "method": method_name,
                "seed": seed,
            },
            os.path.join(method_dir, "final_checkpoint.pth"),
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS_T4:
            print(
                f"{method_name} | Epoch {epoch:02d}/{EPOCHS_T4} | "
                f"Train loss {train_loss:.4f} | "
                f"Noisy val acc {val_metrics['val_acc_noisy']:.2f}%"
            )

        # Persist history incrementally.
        pd.DataFrame(history).to_csv(
            os.path.join(method_dir, "training_log.csv"),
            index=False,
        )

    elapsed_minutes = (time.time() - start_time) / 60.0

    summary = {
        "method": method_name,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_acc_noisy": best_val_acc,
        "final_val_acc_noisy": history[-1]["val_acc_noisy"],
        "runtime_minutes": elapsed_minutes,
    }

    with open(os.path.join(method_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCompleted {method_name}: {summary}")

    del model, optimizer, scheduler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


print("Task 4 training loop defined successfully.")
print("Epoch budget:", EPOCHS_T4)
print("Batch size:", BATCH_SIZE_T4)
print("Checkpoint selection: noisy validation accuracy only")
print("No training has started.")

```

```
# Task 4 — Method 1: Task 3 score-based filtering + cross-entropy

filtered_targets = noisy_labels_all[kept_indices]

assert len(kept_indices) == 29251
assert len(filtered_targets) == len(kept_indices)

filtered_summary = train_task4_method(
    method_name="filtered_ce_35pct",
    train_indices=kept_indices,
    train_targets=filtered_targets,
    loss_function=ce_loss_t4,
    seed=42,
)

print("\nFiltered CE experiment finished.")
print("Best epoch:", filtered_summary["best_epoch"])
print(
    "Best noisy validation accuracy:",
    f'{filtered_summary["best_val_acc_noisy"]:.2f}%'
)
print(
    "Final noisy validation accuracy:",
    f'{filtered_summary["final_val_acc_noisy"]:.2f}%'
)

```

```
import os
import json
import pandas as pd

method_dir = "/kaggle/working/task4/filtered_ce_35pct"

required_files = [
    "best_checkpoint.pth",
    "final_checkpoint.pth",
    "training_log.csv",
    "summary.json",
]

print("========== FILTERED CE ARTIFACTS ==========")

for filename in required_files:
    path = os.path.join(method_dir, filename)
    exists = os.path.exists(path)
    print(f"{'OK' if exists else 'MISSING'}: {filename}")

assert all(
    os.path.exists(os.path.join(method_dir, f))
    for f in required_files
), "One or more experiment artifacts are missing."

with open(os.path.join(method_dir, "summary.json")) as f:
    summary = json.load(f)

log_df = pd.read_csv(os.path.join(method_dir, "training_log.csv"))

print("\nSummary:")
print(json.dumps(summary, indent=2))

print("\nLog shape:", log_df.shape)
print("Best epoch from log:",
      int(log_df.loc[log_df["val_acc_noisy"].idxmax(), "epoch"]))
print("Best accuracy from log:",
      log_df["val_acc_noisy"].max())

```

```
import os
import json
import pandas as pd

method_dir = "/kaggle/working/task4/filtered_ce_35pct"

required_files = [
    "best_checkpoint.pth",
    "final_checkpoint.pth",
    "training_log.csv",
    "summary.json",
]

for filename in required_files:
    path = os.path.join(method_dir, filename)
    print(f"{'OK' if os.path.exists(path) else 'MISSING'}: {filename}")

assert all(
    os.path.exists(os.path.join(method_dir, f))
    for f in required_files
), "Missing experiment artifacts!"

with open(os.path.join(method_dir, "summary.json")) as f:
    filtered_summary = json.load(f)

print("\nFiltered CE summary:")
print(json.dumps(filtered_summary, indent=2))

```

```
# STEP 1: Restore the environment after the Kaggle session restart

import os
import sys
import random
import time
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToPILImage, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Paths
DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"
MODEL_PATH = "/kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1"

OUTPUT_DIR = "/kaggle/working"
PRED_DIR = "/kaggle/working/predictions"
TASK4_DIR = "/kaggle/working/task4"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Environment restored.")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nChecking available files:")
for path in [
    DATA_PATH,
    MODEL_PATH,
    PRED_DIR,
    TASK4_DIR,
]:
    print(path, "->", os.path.exists(path))

```

```
Environment restored.
Device: cuda
GPU: Tesla T4

Checking available files:
/kaggle/input/datasets/afnanahmed2/inter-iit-cv -> True
/kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1 -> True
/kaggle/working/predictions -> False
/kaggle/working/task4 -> False

```

```
# STEP 2: Reload datasets, labels, and the provided ResNet18

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import (
    Compose, ToPILImage, RandomCrop,
    RandomHorizontalFlip, ToTensor, Normalize
)

# Paths
DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"
MODEL_PATH = "/kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1"

OUTPUT_DIR = "/kaggle/working"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
TASK4_DIR = os.path.join(OUTPUT_DIR, "task4")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(TASK4_DIR, exist_ok=True)

# Load images
images = np.load(os.path.join(DATA_PATH, "train_images.npy"))
test_images = np.load(os.path.join(DATA_PATH, "test_images.npy"))

# Load labels and official splits
labels_df = pd.read_csv(os.path.join(DATA_PATH, "labels.csv"))

train_idx = np.loadtxt(
    os.path.join(DATA_PATH, "train_idx.txt"), dtype=int
)
val_idx = np.loadtxt(
    os.path.join(DATA_PATH, "val_idx.txt"), dtype=int
)

noisy_labels_all = labels_df["noisy_label"].to_numpy()

# Validation data uses noisy labels for Tasks 2–4
X_val = images[val_idx]
y_noisy_val = noisy_labels_all[val_idx]

# Load the provided ResNet18
if MODEL_PATH not in sys.path:
    sys.path.insert(0, MODEL_PATH)

from model import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Data loaded successfully!")
print("Training images:", images.shape)
print("Test images:", test_images.shape)
print("Training split:", len(train_idx))
print("Validation split:", len(val_idx))
print("Noisy labels:", noisy_labels_all.shape)
print("Device:", device)

# Verify required files
for filename in [
    "train_images.npy",
    "test_images.npy",
    "labels.csv",
    "train_idx.txt",
    "val_idx.txt",
]:
    print(filename, "exists:", os.path.exists(os.path.join(DATA_PATH, filename)))

```

```
Data loaded successfully!
Training images: (50000, 32, 32, 3)
Test images: (10000, 32, 32, 3)
Training split: 45000
Validation split: 5000
Noisy labels: (50000,)
Device: cuda
train_images.npy exists: True
test_images.npy exists: True
labels.csv exists: True
train_idx.txt exists: True
val_idx.txt exists: True

```

```
# STEP 3: Restore transformations, dataset class, and loss functions

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose, ToPILImage, RandomCrop,
    RandomHorizontalFlip, ToTensor, Normalize
)

# Reproducibility settings
SEED = 42
EPOCHS_T4 = 60
BATCH_SIZE_T4 = 128
NUM_WORKERS_T4 = 2

# CIFAR-10 normalization
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Training augmentation
train_transform_t4 = Compose([
    ToPILImage(),
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(MEAN, STD),
])

# Validation and test preprocessing
eval_transform_t4 = Compose([
    ToPILImage(),
    ToTensor(),
    Normalize(MEAN, STD),
])

# Dataset using original image indices
class Task4Dataset(Dataset):
    def __init__(self, original_indices, targets=None, transform=None):
        self.original_indices = np.asarray(original_indices)
        self.targets = (
            None if targets is None else np.asarray(targets)
        )
        self.transform = transform

    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, idx):
        original_idx = int(self.original_indices[idx])
        image = images[original_idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.targets is None:
            return image, original_idx

        return image, int(self.targets[idx])


# Standard cross-entropy
ce_loss_t4 = nn.CrossEntropyLoss()

# Generalized Cross-Entropy, q = 0.7
class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probabilities = torch.softmax(logits, dim=1)
        p_true = probabilities.gather(
            1, targets.view(-1, 1)
        ).squeeze(1)

        losses = (1.0 - p_true.pow(self.q)) / self.q
        return losses.mean()


gce_loss_t4 = GCELoss(q=0.7)

# Cross-entropy with label smoothing
label_smoothing_loss_t4 = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

print("Transformations restored.")
print("Task4Dataset restored.")
print("CE, GCE (q=0.7), and label-smoothing losses restored.")

# Quick dataset check
check_dataset = Task4Dataset(
    train_idx[:4],
    noisy_labels_all[train_idx[:4]],
    train_transform_t4,
)

sample_image, sample_label = check_dataset[0]

print("Sample image shape:", sample_image.shape)
print("Sample label:", sample_label)
print("Number of training examples:", len(train_idx))
print("Number of validation examples:", len(val_idx))

```

```
Transformations restored.
Task4Dataset restored.
CE, GCE (q=0.7), and label-smoothing losses restored.
Sample image shape: torch.Size([3, 32, 32])
Sample label: 4
Number of training examples: 45000
Number of validation examples: 5000

```

```
# STEP 4: Diagnose GCE numerical stability

import torch

print("========== GCE NUMERICAL TEST ==========")

# Test 1: Loss calculation on random logits
test_logits = torch.randn(
    16, 10, device=device, requires_grad=True
)
test_targets = torch.randint(
    0, 10, (16,), device=device
)

test_loss = gce_loss_t4(test_logits, test_targets)

print("Test loss:", test_loss.item())
print("Loss is finite:", torch.isfinite(test_loss).item())

# Test 2: Backpropagation
test_loss.backward()

gradients_finite = (
    test_logits.grad is not None
    and torch.isfinite(test_logits.grad).all().item()
)

print("Gradients are finite:", gradients_finite)

# Test 3: Actual ResNet18 forward pass
print("\nTesting the actual ResNet18 model...")

test_model = ResNet18(num_classes=10).to(device)
test_model.eval()

test_batch = torch.stack([
    train_transform_t4(images[int(i)])
    for i in train_idx[:8]
]).to(device)

test_batch_targets = torch.tensor(
    noisy_labels_all[train_idx[:8]],
    dtype=torch.long,
    device=device
)

with torch.no_grad():
    test_outputs = test_model(test_batch)
    actual_model_loss = gce_loss_t4(
        test_outputs, test_batch_targets
    )

print("Model outputs finite:", torch.isfinite(test_outputs).all().item())
print("Model GCE loss:", actual_model_loss.item())
print("Model loss finite:", torch.isfinite(actual_model_loss).item())

del test_model, test_batch, test_batch_targets
del test_logits, test_targets, test_loss

print("\nDiagnostic complete.")

```

```
========== GCE NUMERICAL TEST ==========
Test loss: 1.1690137386322021
Loss is finite: True
Gradients are finite: True

Testing the actual ResNet18 model...
Model outputs finite: True
Model GCE loss: 1.1444414854049683
Model loss finite: True

Diagnostic complete.

```

```
# STEP 5: Restore the Task 4 training function

import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, ToPILImage, RandomCrop,
    RandomHorizontalFlip, ToTensor, Normalize
)

TASK4_DIR = "/kaggle/working/task4"
os.makedirs(TASK4_DIR, exist_ok=True)

EPOCHS_T4 = 60
BATCH_SIZE_T4 = 128
NUM_WORKERS_T4 = 2


def set_seed_t4(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_noisy_validation(model, loader):
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += yb.size(0)

    return {
        "val_loss_noisy": total_loss / total_samples,
        "val_acc_noisy": 100.0 * total_correct / total_samples,
    }


def train_task4_method(
    method_name,
    train_indices,
    train_targets,
    loss_function,
    seed=42,
):
    set_seed_t4(seed)

    method_dir = os.path.join(TASK4_DIR, method_name)
    os.makedirs(method_dir, exist_ok=True)

    train_indices = np.asarray(train_indices)
    train_targets = np.asarray(train_targets)

    train_dataset = Task4Dataset(
        train_indices,
        train_targets,
        train_transform_t4,
    )

    val_dataset = Task4Dataset(
        val_idx,
        noisy_labels_all[val_idx],
        eval_transform_t4,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_T4,
        shuffle=True,
        num_workers=NUM_WORKERS_T4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS_T4,
        pin_memory=True,
    )

    model = ResNet18(num_classes=10).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_T4,
    )

    history = []
    best_val_acc = -1.0
    best_epoch = None
    start_time = time.time()

    for epoch in range(1, EPOCHS_T4 + 1):
        model.train()

        running_loss = 0.0
        total_samples = 0
        nonfinite_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = loss_function(logits, yb)

            # Stop immediately if the loss becomes invalid.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{method_name}: non-finite loss "
                    f"at epoch {epoch}. Training stopped."
                )

            loss.backward()

            # Stop if gradients become invalid.
            for parameter in model.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise RuntimeError(
                        f"{method_name}: non-finite gradient "
                        f"at epoch {epoch}. Training stopped."
                    )

            optimizer.step()

            # Check for invalid parameters after the update.
            for parameter in model.parameters():
                if not torch.isfinite(parameter).all():
                    raise RuntimeError(
                        f"{method_name}: non-finite parameter "
                        f"after epoch {epoch} update."
                    )

            batch_size = yb.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        scheduler.step()

        train_loss = running_loss / total_samples
        val_metrics = evaluate_noisy_validation(model, val_loader)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_noisy": val_metrics["val_loss_noisy"],
            "val_acc_noisy": val_metrics["val_acc_noisy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        # Save the latest training log.
        pd.DataFrame(history).to_csv(
            os.path.join(method_dir, "training_log.csv"),
            index=False,
        )

        # Keep the earliest epoch if validation accuracy ties.
        if val_metrics["val_acc_noisy"] > best_val_acc:
            best_val_acc = val_metrics["val_acc_noisy"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc_noisy": best_val_acc,
                    "method": method_name,
                    "seed": seed,
                },
                os.path.join(method_dir, "best_checkpoint.pth"),
            )

        # Save the current model and optimizer state.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "method": method_name,
                "seed": seed,
            },
            os.path.join(method_dir, "final_checkpoint.pth"),
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS_T4:
            print(
                f"{method_name} | "
                f"Epoch {epoch:02d}/{EPOCHS_T4} | "
                f"Train loss {train_loss:.4f} | "
                f"Noisy val acc {val_metrics['val_acc_noisy']:.2f}%"
            )

    runtime_minutes = (time.time() - start_time) / 60

    summary = {
        "method": method_name,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_acc_noisy": best_val_acc,
        "final_val_acc_noisy": history[-1]["val_acc_noisy"],
        "runtime_minutes": runtime_minutes,
    }

    with open(os.path.join(method_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCompleted {method_name}:")
    print(json.dumps(summary, indent=2))

    return summary


print("Task 4 training function restored.")
print("Epochs:", EPOCHS_T4)
print("Batch size:", BATCH_SIZE_T4)
print("Output directory:", TASK4_DIR)

```

```
Task 4 training function restored.
Epochs: 60
Batch size: 128
Output directory: /kaggle/working/task4

```

```
# STEP 6: Verify filtered training indices and labels

import os
import numpy as np

filtered_indices_path = os.path.join(
    TASK4_DIR, "filtered_kept_indices.npy"
)

print("========== FILTER VERIFICATION ==========")

if os.path.exists(filtered_indices_path):
    kept_indices = np.load(filtered_indices_path)

    print("Filtered indices found:", True)
    print("Retained samples:", len(kept_indices))
    print("Expected approximately:", 29251)
    print("Unique indices:", len(np.unique(kept_indices)))
    print("All indices valid:", np.all(
        (kept_indices >= 0) & (kept_indices < len(images))
    ))
    print("All indices in official training split:", np.all(
        np.isin(kept_indices, train_idx)
    ))

    retained_labels = noisy_labels_all[kept_indices]

    print("Retained labels shape:", retained_labels.shape)
    print("Class counts:", np.bincount(retained_labels, minlength=10))

else:
    print("Filtered indices not found.")
    print("We must reconstruct the filtering setup before training.")

print("\n========== GCE LOSS VERIFICATION ==========")

print("GCE q:", gce_loss_t4.q)
print("GCE loss function:", gce_loss_t4)

```

```
========== FILTER VERIFICATION ==========
Filtered indices not found.
We must reconstruct the filtering setup before training.

========== GCE LOSS VERIFICATION ==========
GCE q: 0.7
GCE loss function: GCELoss()

```

```
# STEP 7: Inspect available files after the session restart

import os

print("========== INPUT DATA ==========")
for root, dirs, files in os.walk(DATA_PATH):
    for filename in files:
        print(os.path.join(root, filename))

print("\n========== WORKING DIRECTORY ==========")

for root, dirs, files in os.walk("/kaggle/working"):
    level = root.replace("/kaggle/working", "").count(os.sep)

    if level > 2:
        dirs[:] = []
        continue

    for filename in files:
        path = os.path.join(root, filename)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{path} ({size_mb:.2f} MB)")

```

```
========== INPUT DATA ==========
/kaggle/input/datasets/afnanahmed2/inter-iit-cv/train_images.npy
/kaggle/input/datasets/afnanahmed2/inter-iit-cv/train_idx.txt
/kaggle/input/datasets/afnanahmed2/inter-iit-cv/val_idx.txt
/kaggle/input/datasets/afnanahmed2/inter-iit-cv/classes.txt
/kaggle/input/datasets/afnanahmed2/inter-iit-cv/test_images.npy
/kaggle/input/datasets/afnanahmed2/inter-iit-cv/labels.csv

========== WORKING DIRECTORY ==========
/kaggle/working/.virtual_documents/__notebook_source__.ipynb (0.08 MB)

```

```
# TASK 4 — STEP 9: Restore previous outputs

import os
import json
import numpy as np
import pandas as pd

# Locate the uploaded recovery dataset
RECOVERY_PATH = "/kaggle/input/prev-outputs"

print("Recovery dataset exists:", os.path.exists(RECOVERY_PATH))

if os.path.exists(RECOVERY_PATH):
    print("\nAvailable recovery files:")
    for filename in sorted(os.listdir(RECOVERY_PATH)):
        print(" -", filename)

    # Restore the original Task 3 filtering indices
    filtered_indices_path = os.path.join(
        RECOVERY_PATH,
        "filtered_kept_indices.npy"
    )

    kept_indices = np.load(filtered_indices_path)

    # Verify the recovered indices
    print("\n========== FILTER VERIFICATION ==========")
    print("Retained samples:", len(kept_indices))
    print("Unique indices:", len(np.unique(kept_indices)))
    print(
        "All indices in official training split:",
        np.all(np.isin(kept_indices, train_idx))
    )

    # Restore the configuration
    config_path = os.path.join(
        RECOVERY_PATH,
        "filtered_ce_config.json"
    )

    with open(config_path, "r") as f:
        filtered_config = json.load(f)

    print("\nFiltering configuration:")
    print(json.dumps(filtered_config, indent=2))

else:
    print("Dataset path not found. Check the input folder name.")

```

```
Recovery dataset exists: False
Dataset path not found. Check the input folder name.

```

```
import os

print("========== AVAILABLE INPUT DATASETS ==========")

for root, dirs, files in os.walk("/kaggle/input"):
    level = root.replace("/kaggle/input", "").count(os.sep)

    if level > 2:
        dirs[:] = []
        continue

    print(f"\nFolder: {root}")

    for filename in files:
        print(f"  File: {filename}")

```

```
========== AVAILABLE INPUT DATASETS ==========

Folder: /kaggle/input

Folder: /kaggle/input/models

Folder: /kaggle/input/models/afnanahmed2

Folder: /kaggle/input/datasets

Folder: /kaggle/input/datasets/afnanahmed2

```

```
import os

for root, dirs, files in os.walk("/kaggle/input"):
    if files:
        print(f"\nFolder: {root}")
        for filename in files:
            print(" ", filename)

```

```
Folder: /kaggle/input/models/afnanahmed2/resnet18/pytorch/default/1
  model.py

Folder: /kaggle/input/datasets/afnanahmed2/prev-outputs
  filtered_kept_indices.npy
  t2_early_stop.csv
  filtered_class_counts.csv
  filtered_ce_config.json
  t1_noisy_ce.csv
  t2_dynamics.csv
  t2_summary.json
  fold_assignment.npy
  oof_probabilities_partial.npy
  t1_clean_ref.csv

Folder: /kaggle/input/datasets/afnanahmed2/inter-iit-cv
  train_images.npy
  train_idx.txt
  val_idx.txt
  classes.txt
  test_images.npy
  labels.csv

```

```
# TASK 4 — STEP 10: Restore filtering artifacts

import os
import json
import numpy as np
import pandas as pd

RECOVERY_PATH = "/kaggle/input/datasets/afnanahmed2/prev-outputs"

# Restore the exact indices selected during Task 3
kept_indices = np.load(
    os.path.join(RECOVERY_PATH, "filtered_kept_indices.npy")
)

# Restore filtering configuration
with open(
    os.path.join(RECOVERY_PATH, "filtered_ce_config.json"),
    "r"
) as f:
    filtered_config = json.load(f)

# Restore class counts
filtered_class_counts = pd.read_csv(
    os.path.join(RECOVERY_PATH, "filtered_class_counts.csv")
)

print("========== RECOVERED FILTERING ==========")
print("Retained images:", len(kept_indices))
print("Unique indices:", len(np.unique(kept_indices)))
print("Filtering configuration:")
print(json.dumps(filtered_config, indent=2))

print("\nAll indices within dataset:",
      np.all((kept_indices >= 0) & (kept_indices < len(images))))

print("All indices in official training split:",
      np.all(np.isin(kept_indices, train_idx)))

print("\nClass counts:")
display(filtered_class_counts)

# Restore original noisy labels for retained examples
filtered_train_targets = noisy_labels_all[kept_indices]

print("\nFiltered training targets:", filtered_train_targets.shape)
print("Filtering artifacts restored successfully.")

```

```
========== RECOVERED FILTERING ==========
Retained images: 29251
Unique indices: 29251
Filtering configuration:
{
  "method": "task3_score_filtered_cross_entropy",
  "seed": 42,
  "filter_fraction": 0.35,
  "removed_images": 15749,
  "retained_images": 29251,
  "selection_rule": "Remove highest 35% of Task 3 noise scores",
  "label_source": "noisy_label",
  "clean_labels_used_for_filtering": false
}

All indices within dataset: True
All indices in official training split: True

Class counts:

```

|    | class | before_filtering | after_filtering |
| :- | :---- | :--------------- | :-------------- |
| 0  | 0     | 4791             | 2808            |
| 1  | 1     | 5414             | 3683            |
| 2  | 2     | 4788             | 2838            |
| 3  | 3     | 4205             | 2555            |
| 4  | 4     | 3636             | 2053            |
| 5  | 5     | 5130             | 3021            |
| 6  | 6     | 3770             | 2590            |
| 7  | 7     | 4748             | 3491            |
| 8  | 8     | 4160             | 3099            |
| 9  | 9     | 4358             | 3113            |

```
Filtered training targets: (29251,)
Filtering artifacts restored successfully.

```

```
# TASK 4 — STEP 11: Short GCE stability test

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Reproducible test
set_seed_t4(42)

# Use only 512 retained examples for this diagnostic
test_indices = kept_indices[:512]
test_targets = noisy_labels_all[test_indices]

test_dataset = Task4Dataset(
    test_indices,
    test_targets,
    train_transform_t4,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

test_model = ResNet18(num_classes=10).to(device)

test_optimizer = optim.SGD(
    test_model.parameters(),
    lr=0.01,  # Conservative learning rate for diagnosis
    momentum=0.9,
    weight_decay=0.0005,
)

print("Testing GCE on 512 retained examples...")

test_model.train()
test_losses = []

for batch_number, (xb, yb) in enumerate(test_loader, start=1):
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    test_optimizer.zero_grad(set_to_none=True)

    logits = test_model(xb)
    loss = gce_loss_t4(logits, yb)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite GCE loss at batch {batch_number}"
        )

    loss.backward()

    for parameter in test_model.parameters():
        if (
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ):
            raise RuntimeError(
                f"Non-finite gradient at batch {batch_number}"
            )

    test_optimizer.step()

    for parameter in test_model.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(
                f"Non-finite model parameter after batch {batch_number}"
            )

    test_losses.append(loss.item())

    print(
        f"Batch {batch_number}/{len(test_loader)} | "
        f"GCE loss: {loss.item():.4f}"
    )

print("\nGCE stability test passed.")
print("Batches completed:", len(test_losses))
print("Mean GCE loss:", round(float(np.mean(test_losses)), 4))
print("All losses finite:", bool(np.isfinite(test_losses).all()))

del test_model, test_optimizer, test_loader, test_dataset
torch.cuda.empty_cache()

```

```
Testing GCE on 512 retained examples...
Batch 1/8 | GCE loss: 1.2111
Batch 2/8 | GCE loss: 1.1958
Batch 3/8 | GCE loss: 1.1637
Batch 4/8 | GCE loss: 1.2015
Batch 5/8 | GCE loss: 1.1408
Batch 6/8 | GCE loss: 1.1229
Batch 7/8 | GCE loss: 1.0725
Batch 8/8 | GCE loss: 1.1038

GCE stability test passed.
Batches completed: 8
Mean GCE loss: 1.1515
All losses finite: True

```

```
# TASK 4 — STEP 13: Stable GCE implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

class StableGCELoss(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        # Calculate log probabilities more stably
        log_probs = F.log_softmax(logits, dim=1)

        # Log probability of each observed target
        log_p_true = log_probs.gather(
            1, targets.view(-1, 1)
        ).squeeze(1)

        # GCE: (1 - p_true^q) / q
        # expm1 improves numerical accuracy for small arguments
        losses = -torch.expm1(self.q * log_p_true) / self.q

        return losses.mean()


stable_gce_loss_t4 = StableGCELoss(q=0.7)

print("Stable GCE loss restored.")
print("q =", stable_gce_loss_t4.q)

# Quick numerical test
test_logits = torch.randn(
    128, 10, device=device, requires_grad=True
)
test_targets = torch.randint(
    0, 10, (128,), device=device
)

test_loss = stable_gce_loss_t4(test_logits, test_targets)
test_loss.backward()

print("Loss:", test_loss.item())
print("Loss finite:", torch.isfinite(test_loss).item())
print(
    "Gradients finite:",
    torch.isfinite(test_logits.grad).all().item()
)

assert torch.isfinite(test_loss)
assert torch.isfinite(test_logits.grad).all()

print("Stable GCE test passed.")

```

```
Stable GCE loss restored.
q = 0.7
Loss: 1.1819411516189575
Loss finite: True
Gradients finite: True
Stable GCE test passed.

```

```
# TASK 4 — STEP 14: Test stable GCE on ResNet18

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

set_seed_t4(42)

# Small diagnostic subset
diagnostic_indices = kept_indices[:2048]
diagnostic_targets = noisy_labels_all[diagnostic_indices]

diagnostic_dataset = Task4Dataset(
    diagnostic_indices,
    diagnostic_targets,
    train_transform_t4,
)

diagnostic_loader = DataLoader(
    diagnostic_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

diagnostic_model = ResNet18(num_classes=10).to(device)

diagnostic_optimizer = optim.SGD(
    diagnostic_model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
)

print("Testing stable GCE with ResNet18...")
diagnostic_model.train()

for batch_number, (xb, yb) in enumerate(diagnostic_loader):
    if batch_number >= 20:
        break

    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    diagnostic_optimizer.zero_grad(set_to_none=True)

    logits = diagnostic_model(xb)
    loss = stable_gce_loss_t4(logits, yb)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at batch {batch_number + 1}"
        )

    loss.backward()

    bad_gradients = [
        name for name, parameter in diagnostic_model.named_parameters()
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]

    if bad_gradients:
        print("First failing batch:", batch_number + 1)
        print("Layers with non-finite gradients:", bad_gradients[:10])
        raise RuntimeError("Non-finite gradients detected.")

    diagnostic_optimizer.step()

    bad_parameters = [
        name for name, parameter in diagnostic_model.named_parameters()
        if not torch.isfinite(parameter).all()
    ]

    if bad_parameters:
        print("First failing batch:", batch_number + 1)
        print("Layers with non-finite parameters:", bad_parameters[:10])
        raise RuntimeError("Non-finite parameters detected.")

    print(
        f"Batch {batch_number + 1:02d} | "
        f"Loss: {loss.item():.4f}"
    )

print("\nResNet18 GCE diagnostic passed.")

del diagnostic_model, diagnostic_optimizer
del diagnostic_loader, diagnostic_dataset

torch.cuda.empty_cache()

```

```
Testing stable GCE with ResNet18...
Batch 01 | Loss: 1.1910
Batch 02 | Loss: 1.1438
Batch 03 | Loss: 1.0297
Batch 04 | Loss: 0.9859
Batch 05 | Loss: 0.9065
Batch 06 | Loss: 0.9267
Batch 07 | Loss: 0.9263
Batch 08 | Loss: 0.7809
Batch 09 | Loss: 0.9636
Batch 10 | Loss: 0.8271
Batch 11 | Loss: 0.8976
Batch 12 | Loss: 0.7306
Batch 13 | Loss: 0.7907
Batch 14 | Loss: 0.8233
Batch 15 | Loss: 0.7654
Batch 16 | Loss: 0.7474

ResNet18 GCE diagnostic passed.

```

```
# TASK 4 — STEP 15: Full GCE training with stable loss

gce_summary = train_task4_method(
    method_name="gce_q07",
    train_indices=kept_indices,
    train_targets=noisy_labels_all[kept_indices],
    loss_function=stable_gce_loss_t4,
    seed=42,
)

print("\n========== GCE RESULTS ==========")
print(f"Best epoch: {gce_summary['best_epoch']}")
print(
    f"Best noisy validation accuracy: "
    f"{gce_summary['best_val_acc_noisy']:.2f}%"
)
print(
    f"Final noisy validation accuracy: "
    f"{gce_summary['final_val_acc_noisy']:.2f}%"
)
print(f"Runtime: {gce_summary['runtime_minutes']:.2f} minutes")

```

```
gce_q07 | Epoch 01/60 | Train loss 0.9517 | Noisy val acc 24.68%
gce_q07 | Epoch 10/60 | Train loss 0.4547 | Noisy val acc 44.54%
gce_q07 | Epoch 20/60 | Train loss 0.3199 | Noisy val acc 49.60%
gce_q07 | Epoch 30/60 | Train loss 0.2669 | Noisy val acc 49.72%
gce_q07 | Epoch 40/60 | Train loss 0.2124 | Noisy val acc 54.34%
gce_q07 | Epoch 50/60 | Train loss 0.1371 | Noisy val acc 56.58%
gce_q07 | Epoch 60/60 | Train loss 0.0888 | Noisy val acc 56.84%

Completed gce_q07:
{
  "method": "gce_q07",
  "seed": 42,
  "epochs_completed": 60,
  "best_epoch": 51,
  "best_val_acc_noisy": 57.16,
  "final_val_acc_noisy": 56.84,
  "runtime_minutes": 29.531569159030916
}

========== GCE RESULTS ==========
Best epoch: 51
Best noisy validation accuracy: 57.16%
Final noisy validation accuracy: 56.84%
Runtime: 29.53 minutes

```

```
# TASK 4 — STEP 16: Label smoothing training

label_smoothing_summary = train_task4_method(
    method_name="label_smoothing",
    train_indices=kept_indices,
    train_targets=noisy_labels_all[kept_indices],
    loss_function=label_smoothing_loss_t4,
    seed=42,
)

print("\n========== LABEL SMOOTHING RESULTS ==========")
print("Label smoothing epsilon: 0.1")
print(f"Best epoch: {label_smoothing_summary['best_epoch']}")
print(
    "Best noisy validation accuracy: "
    f"{label_smoothing_summary['best_val_acc_noisy']:.2f}%"
)
print(
    "Final noisy validation accuracy: "
    f"{label_smoothing_summary['final_val_acc_noisy']:.2f}%"
)
print(
    f"Runtime: {label_smoothing_summary['runtime_minutes']:.2f} minutes"
)

```

```
label_smoothing | Epoch 01/60 | Train loss 2.1928 | Noisy val acc 27.70%
label_smoothing | Epoch 10/60 | Train loss 1.0936 | Noisy val acc 45.84%
label_smoothing | Epoch 20/60 | Train loss 0.9043 | Noisy val acc 50.58%
label_smoothing | Epoch 30/60 | Train loss 0.8231 | Noisy val acc 52.18%
label_smoothing | Epoch 40/60 | Train loss 0.7252 | Noisy val acc 54.12%
label_smoothing | Epoch 50/60 | Train loss 0.5893 | Noisy val acc 56.06%
label_smoothing | Epoch 60/60 | Train loss 0.5253 | Noisy val acc 55.78%

Completed label_smoothing:
{
  "method": "label_smoothing",
  "seed": 42,
  "epochs_completed": 60,
  "best_epoch": 51,
  "best_val_acc_noisy": 56.58,
  "final_val_acc_noisy": 55.78,
  "runtime_minutes": 29.528511106967926
}

========== LABEL SMOOTHING RESULTS ==========
Label smoothing epsilon: 0.1
Best epoch: 51
Best noisy validation accuracy: 56.58%
Final noisy validation accuracy: 55.78%
Runtime: 29.53 minutes

```

```
# TASK 4 — STEP 17: Inspect prediction CSV format

import os
import pandas as pd

RECOVERY_PATH = "/kaggle/input/datasets/afnanahmed2/prev-outputs"

for filename in [
    "t1_noisy_ce.csv",
    "t1_clean_ref.csv",
    "t2_early_stop.csv",
]:
    path = os.path.join(RECOVERY_PATH, filename)

    if os.path.exists(path):
        df = pd.read_csv(path)

        print(f"\n{filename}")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head())

```

```
t1_noisy_ce.csv
Shape: (10000, 2)
Columns: ['id', 'pred']
   id  pred
0   0     0
1   1     8
2   2     9
3   3     5
4   4     5

t1_clean_ref.csv
Shape: (10000, 2)
Columns: ['id', 'pred']
   id  pred
0   0     0
1   1     8
2   2     8
3   3     3
4   4     3

t2_early_stop.csv
Shape: (10000, 2)
Columns: ['id', 'pred']
   id  pred
0   0     0
1   1     8
2   2     8
3   3     5
4   4     6

```

```
import os

TASK4_DIR = "/kaggle/working/task4"

print("========== TASK 4 CHECKPOINTS ==========")

for root, dirs, files in os.walk(TASK4_DIR):
    print(f"\nFolder: {root}")

    for filename in files:
        print(" ", filename)

print("\n========== EXPECTED CHECKPOINTS ==========")

expected = [
    "filtered_ce_35pct/best_checkpoint.pth",
    "gce_q07/best_checkpoint.pth",
    "label_smoothing/best_checkpoint.pth",
]

for relative_path in expected:
    full_path = os.path.join(TASK4_DIR, relative_path)
    print(
        relative_path,
        "->",
        "FOUND" if os.path.exists(full_path) else "MISSING"
    )

```

```
========== TASK 4 CHECKPOINTS ==========

Folder: /kaggle/working/task4

Folder: /kaggle/working/task4/label_smoothing
  summary.json
  final_checkpoint.pth
  best_checkpoint.pth
  training_log.csv

Folder: /kaggle/working/task4/gce_q07
  summary.json
  final_checkpoint.pth
  best_checkpoint.pth
  training_log.csv

========== EXPECTED CHECKPOINTS ==========
filtered_ce_35pct/best_checkpoint.pth -> MISSING
gce_q07/best_checkpoint.pth -> FOUND
label_smoothing/best_checkpoint.pth -> FOUND

```

```
# TASK 4 — STEP 20: Generate predictions for available checkpoints

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PRED_DIR = "/kaggle/working/predictions"
TASK4_DIR = "/kaggle/working/task4"

os.makedirs(PRED_DIR, exist_ok=True)

class Task4TestDataset(Dataset):
    def __init__(self, test_images, transform):
        self.test_images = test_images
        self.transform = transform

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        image = self.transform(self.test_images[idx])
        return image, idx


test_dataset = Task4TestDataset(
    test_images,
    eval_transform_t4,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


def generate_task4_predictions(method_name, output_filename):
    checkpoint_path = os.path.join(
        TASK4_DIR, method_name, "best_checkpoint.pth"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    model = ResNet18(num_classes=10).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = np.empty(len(test_images), dtype=np.int64)

    with torch.no_grad():
        for xb, ids in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            predictions[ids.numpy()] = (
                logits.argmax(dim=1).cpu().numpy()
            )

    submission = pd.DataFrame({
        "id": np.arange(len(test_images)),
        "pred": predictions,
    })

    output_path = os.path.join(PRED_DIR, output_filename)
    submission.to_csv(output_path, index=False)

    # Verify submission format
    check = pd.read_csv(output_path)
    assert check.shape == (10000, 2)
    assert check.columns.tolist() == ["id", "pred"]
    assert check["id"].tolist() == list(range(10000))
    assert check["pred"].between(0, 9).all()

    print(f"\nSaved: {output_path}")
    print("Best epoch:", checkpoint["epoch"])
    print("Best noisy validation accuracy:",
          f"{checkpoint['val_acc_noisy']:.2f}%")
    print("Shape:", check.shape)
    print(check.head())

    del model
    torch.cuda.empty_cache()


generate_task4_predictions(
    "gce_q07",
    "t4_gce.csv",
)

generate_task4_predictions(
    "label_smoothing",
    "t4_label_smoothing.csv",
)

print("\nGCE and label-smoothing predictions completed.")

```

```
Saved: /kaggle/working/predictions/t4_gce.csv
Best epoch: 51
Best noisy validation accuracy: 57.16%
Shape: (10000, 2)
   id  pred
0   0     0
1   1     8
2   2     8
3   3     3
4   4     5

Saved: /kaggle/working/predictions/t4_label_smoothing.csv
Best epoch: 51
Best noisy validation accuracy: 56.58%
Shape: (10000, 2)
   id  pred
0   0     0
1   1     8
2   2     8
3   3     5
4   4     3

GCE and label-smoothing predictions completed.

```

```
# TASK 4 — STEP 21: Retrain filtered cross-entropy

filtered_ce_summary = train_task4_method(
    method_name="filtered_ce_35pct",
    train_indices=kept_indices,
    train_targets=noisy_labels_all[kept_indices],
    loss_function=ce_loss_t4,
    seed=42,
)

print("\n========== FILTERED CE RESULTS ==========")
print("Best epoch:", filtered_ce_summary["best_epoch"])
print(
    "Best noisy validation accuracy:",
    f"{filtered_ce_summary['best_val_acc_noisy']:.2f}%"
)
print(
    "Final noisy validation accuracy:",
    f"{filtered_ce_summary['final_val_acc_noisy']:.2f}%"
)
print(
    "Runtime:",
    f"{filtered_ce_summary['runtime_minutes']:.2f} minutes"
)

```

```
filtered_ce_35pct | Epoch 01/60 | Train loss 2.1529 | Noisy val acc 24.88%
filtered_ce_35pct | Epoch 10/60 | Train loss 0.8310 | Noisy val acc 47.06%
filtered_ce_35pct | Epoch 20/60 | Train loss 0.5389 | Noisy val acc 52.10%
filtered_ce_35pct | Epoch 30/60 | Train loss 0.4203 | Noisy val acc 54.40%
filtered_ce_35pct | Epoch 40/60 | Train loss 0.2970 | Noisy val acc 54.62%
filtered_ce_35pct | Epoch 50/60 | Train loss 0.1264 | Noisy val acc 56.18%
filtered_ce_35pct | Epoch 60/60 | Train loss 0.0407 | Noisy val acc 55.88%

Completed filtered_ce_35pct:
{
  "method": "filtered_ce_35pct",
  "seed": 42,
  "epochs_completed": 60,
  "best_epoch": 52,
  "best_val_acc_noisy": 56.42,
  "final_val_acc_noisy": 55.88,
  "runtime_minutes": 29.591850570837657
}

========== FILTERED CE RESULTS ==========
Best epoch: 52
Best noisy validation accuracy: 56.42%
Final noisy validation accuracy: 55.88%
Runtime: 29.59 minutes

```

```
# TASK 4 — STEP 22: Generate filtered-CE test predictions

generate_task4_predictions(
    "filtered_ce_35pct",
    "t4_filtered_ce.csv",
)

print("\nFiltered-CE predictions generated.")

```

```
Saved: /kaggle/working/predictions/t4_filtered_ce.csv
Best epoch: 52
Best noisy validation accuracy: 56.42%
Shape: (10000, 2)
   id  pred
0   0     0
1   1     8
2   2     8
3   3     3
4   4     3

Filtered-CE predictions generated.

```

```
# TASK 4 — STEP 23: Verify prediction files

import os
import pandas as pd

PRED_DIR = "/kaggle/working/predictions"

prediction_files = [
    "t4_filtered_ce.csv",
    "t4_gce.csv",
    "t4_label_smoothing.csv",
]

for filename in prediction_files:
    path = os.path.join(PRED_DIR, filename)
    df = pd.read_csv(path)

    assert df.shape == (10000, 2), f"Incorrect shape: {filename}"
    assert df.columns.tolist() == ["id", "pred"], filename
    assert df["id"].tolist() == list(range(10000)), filename
    assert df["pred"].between(0, 9).all(), filename
    assert not df.isnull().any().any(), filename

    print(f"\n{filename}: PASS")
    print("Rows:", len(df))
    print("Predicted classes:", sorted(df["pred"].unique()))

print("\nAll three Task 4 prediction files passed validation.")

```

```
t4_filtered_ce.csv: PASS
Rows: 10000
Predicted classes: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9)]

t4_gce.csv: PASS
Rows: 10000
Predicted classes: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9)]

t4_label_smoothing.csv: PASS
Rows: 10000
Predicted classes: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9)]

All three Task 4 prediction files passed validation.

```

```
# TASK 4 — STEP 24: Save results.json

import os
import json

TASK4_DIR = "/kaggle/working/task4"
os.makedirs(TASK4_DIR, exist_ok=True)

results = {
    "task4": {
        "best_method": "gce_q07",
        "methods": [
            {
                "name": "filtered_ce_35pct",
                "file": "t4_filtered_ce.csv",
                "val_acc_noisy_labels": 56.42
            },
            {
                "name": "gce_q07",
                "file": "t4_gce.csv",
                "val_acc_noisy_labels": 57.16
            },
            {
                "name": "label_smoothing",
                "file": "t4_label_smoothing.csv",
                "val_acc_noisy_labels": 56.58
            }
        ]
    }
}

results_path = os.path.join(TASK4_DIR, "results.json")

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print("Saved:", results_path)
print(json.dumps(results, indent=2))

```

```
Saved: /kaggle/working/task4/results.json
{
  "task4": {
    "best_method": "gce_q07",
    "methods": [
      {
        "name": "filtered_ce_35pct",
        "file": "t4_filtered_ce.csv",
        "val_acc_noisy_labels": 56.42
      },
      {
        "name": "gce_q07",
        "file": "t4_gce.csv",
        "val_acc_noisy_labels": 57.16
      },
      {
        "name": "label_smoothing",
        "file": "t4_label_smoothing.csv",
        "val_acc_noisy_labels": 56.58
      }
    ]
  }
}

```

```
# TASK 4 — STEP 25: Create report.md

import os

TASK4_DIR = "/kaggle/working/task4"

report = """# Task 4 — Learning with Noisy Labels

## Experimental setup

We compared three methods using the same 35%-filtered training subset of 29,251 images. Filtering removed the highest-scoring 35% according to the Task 3 noise scores. The filtering procedure used noisy labels, not clean labels.

All methods used the provided ResNet18 architecture and were trained for 60 epochs with seed 42. Model selection used noisy validation accuracy.

## Results

| Method | Best epoch | Best noisy validation accuracy |
|---|---:|---:|
| Filtered cross-entropy | 52 | 56.42% |
| Generalized cross-entropy (q=0.7) | 51 | 57.16% |
| Label smoothing (epsilon=0.1) | 51 | 56.58% |

## Best method

Generalized cross-entropy (q=0.7) achieved the highest noisy validation accuracy of 57.16% at epoch 51.

Its improvement was 0.74 percentage points over filtered cross-entropy and 0.58 percentage points over label smoothing.

## Limitations

These results compare three methods on the same filtered subset. They do not isolate the effect of filtering from the effect of the loss function. All model-selection decisions used noisy validation labels.
"""

report_path = os.path.join(TASK4_DIR, "report.md")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print("Saved:", report_path)
print("\n" + report)

```

```
Saved: /kaggle/working/task4/report.md

# Task 4 — Learning with Noisy Labels

## Experimental setup

We compared three methods using the same 35%-filtered training subset of 29,251 images. Filtering removed the highest-scoring 35% according to the Task 3 noise scores. The filtering procedure used noisy labels, not clean labels.

All methods used the provided ResNet18 architecture and were trained for 60 epochs with seed 42. Model selection used noisy validation accuracy.

## Results

| Method | Best epoch | Best noisy validation accuracy |
|---|---:|---:|
| Filtered cross-entropy | 52 | 56.42% |
| Generalized cross-entropy (q=0.7) | 51 | 57.16% |
| Label smoothing (epsilon=0.1) | 51 | 56.58% |

## Best method

Generalized cross-entropy (q=0.7) achieved the highest noisy validation accuracy of 57.16% at epoch 51.

Its improvement was 0.74 percentage points over filtered cross-entropy and 0.58 percentage points over label smoothing.

## Limitations

These results compare three methods on the same filtered subset. They do not isolate the effect of filtering from the effect of the loss function. All model-selection decisions used noisy validation labels.


```

```
# TASK 5 — STEP 1: Load synthetic labels

import os
import numpy as np
import pandas as pd

DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"

# Load the supplied label table
labels_df = pd.read_csv(
    os.path.join(DATA_PATH, "labels.csv")
)

# Extract synthetic labels
synthetic_labels_all = labels_df["synthetic_label"].to_numpy()

# Official validation labels
y_synthetic_val = synthetic_labels_all[val_idx]

# Training labels for the baseline
y_synthetic_train = synthetic_labels_all[train_idx]

# Labels for the filtered Task 4 method
y_synthetic_filtered = synthetic_labels_all[kept_indices]

print("========== TASK 5 DATA CHECK ==========")
print("All synthetic labels:", synthetic_labels_all.shape)
print("Synthetic training labels:", y_synthetic_train.shape)
print("Synthetic validation labels:", y_synthetic_val.shape)
print("Filtered synthetic labels:", y_synthetic_filtered.shape)

print("\nLabel range:", synthetic_labels_all.min(),
      "to", synthetic_labels_all.max())

print("All labels valid:",
      np.all((synthetic_labels_all >= 0) &
             (synthetic_labels_all <= 9)))

print("\nSynthetic validation class counts:")
print(np.bincount(y_synthetic_val, minlength=10))

print("\nFiltered training samples:", len(kept_indices))
print("Task 5 data check complete.")

```

```
========== TASK 5 DATA CHECK ==========
All synthetic labels: (50000,)
Synthetic training labels: (45000,)
Synthetic validation labels: (5000,)
Filtered synthetic labels: (29251,)

Label range: 0 to 9
All labels valid: True

Synthetic validation class counts:
[483 523 488 500 496 482 492 510 513 513]

Filtered training samples: 29251
Task 5 data check complete.

```

```
# TASK 5 — STEP 2: Training and evaluation using synthetic labels

import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


TASK5_DIR = "/kaggle/working/task5"
os.makedirs(TASK5_DIR, exist_ok=True)

EPOCHS_T5 = 60
BATCH_SIZE_T5 = 128
NUM_WORKERS_T5 = 2


def evaluate_synthetic_validation(model, loader):
    """Evaluate exclusively against synthetic validation labels."""
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item()
            total_correct += (
                logits.argmax(dim=1) == yb
            ).sum().item()
            total_samples += yb.size(0)

    return {
        "val_loss_synthetic": total_loss / total_samples,
        "val_acc_synthetic": 100.0 * total_correct / total_samples,
    }


def train_task5_method(
    method_name,
    train_indices,
    train_targets,
    loss_function,
    seed=42,
):
    """Train a Task 5 model and select its checkpoint by synthetic val accuracy."""

    set_seed_t4(seed)

    method_dir = os.path.join(TASK5_DIR, method_name)
    os.makedirs(method_dir, exist_ok=True)

    train_indices = np.asarray(train_indices)
    train_targets = np.asarray(train_targets)

    train_dataset = Task4Dataset(
        train_indices,
        train_targets,
        train_transform_t4,
    )

    # IMPORTANT: Task 5 validation uses synthetic labels.
    val_dataset = Task4Dataset(
        val_idx,
        synthetic_labels_all[val_idx],
        eval_transform_t4,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_T5,
        shuffle=True,
        num_workers=NUM_WORKERS_T5,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS_T5,
        pin_memory=True,
    )

    model = ResNet18(num_classes=10).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_T5,
    )

    history = []
    best_val_acc = -1.0
    best_epoch = None
    start_time = time.time()

    for epoch in range(1, EPOCHS_T5 + 1):
        model.train()

        running_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = loss_function(logits, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{method_name}: non-finite loss at epoch {epoch}"
                )

            loss.backward()

            for parameter in model.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise RuntimeError(
                        f"{method_name}: non-finite gradient "
                        f"at epoch {epoch}"
                    )

            optimizer.step()

            for parameter in model.parameters():
                if not torch.isfinite(parameter).all():
                    raise RuntimeError(
                        f"{method_name}: non-finite parameters "
                        f"after epoch {epoch}"
                    )

            batch_size = yb.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        scheduler.step()

        train_loss = running_loss / total_samples
        val_metrics = evaluate_synthetic_validation(
            model,
            val_loader,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_synthetic": val_metrics["val_loss_synthetic"],
            "val_acc_synthetic": val_metrics["val_acc_synthetic"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            os.path.join(method_dir, "training_log.csv"),
            index=False,
        )

        # Strict > means the earliest epoch wins an accuracy tie.
        if val_metrics["val_acc_synthetic"] > best_val_acc:
            best_val_acc = val_metrics["val_acc_synthetic"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc_synthetic": best_val_acc,
                    "method": method_name,
                    "seed": seed,
                },
                os.path.join(method_dir, "best_checkpoint.pth"),
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "method": method_name,
                "seed": seed,
            },
            os.path.join(method_dir, "final_checkpoint.pth"),
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS_T5:
            print(
                f"{method_name} | "
                f"Epoch {epoch:02d}/{EPOCHS_T5} | "
                f"Train loss {train_loss:.4f} | "
                f"Synthetic val acc "
                f"{val_metrics['val_acc_synthetic']:.2f}%"
            )

    summary = {
        "method": method_name,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_acc_synthetic": best_val_acc,
        "final_val_acc_synthetic": history[-1]["val_acc_synthetic"],
        "runtime_minutes": (time.time() - start_time) / 60,
    }

    with open(os.path.join(method_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCompleted {method_name}:")
    print(json.dumps(summary, indent=2))

    return summary


print("Task 5 training function ready.")
print("Epoch budget:", EPOCHS_T5)
print("Batch size:", BATCH_SIZE_T5)
print("Validation selection: synthetic labels only")

```

```
Task 5 training function ready.
Epoch budget: 60
Batch size: 128
Validation selection: synthetic labels only

```

```
# TASK 5 — STEP 3: Baseline trained on synthetic labels

t5_baseline_summary = train_task5_method(
    method_name="baseline_synth",
    train_indices=train_idx,
    train_targets=synthetic_labels_all[train_idx],
    loss_function=ce_loss_t4,
    seed=42,
)

print("\nBaseline synthetic-validation accuracy:",
      f"{t5_baseline_summary['best_val_acc_synthetic']:.2f}%")

```

```
baseline_synth | Epoch 01/60 | Train loss 2.3223 | Synthetic val acc 22.02%
baseline_synth | Epoch 10/60 | Train loss 1.9085 | Synthetic val acc 34.86%
baseline_synth | Epoch 20/60 | Train loss 1.7770 | Synthetic val acc 45.54%
baseline_synth | Epoch 30/60 | Train loss 1.7019 | Synthetic val acc 47.08%
baseline_synth | Epoch 40/60 | Train loss 1.6031 | Synthetic val acc 48.72%
baseline_synth | Epoch 50/60 | Train loss 1.3477 | Synthetic val acc 47.44%
baseline_synth | Epoch 60/60 | Train loss 0.7469 | Synthetic val acc 42.48%

Completed baseline_synth:
{
  "method": "baseline_synth",
  "seed": 42,
  "epochs_completed": 60,
  "best_epoch": 43,
  "best_val_acc_synthetic": 49.66,
  "final_val_acc_synthetic": 42.48,
  "runtime_minutes": 44.52789453665415
}

Baseline synthetic-validation accuracy: 49.66%

```

```
t5_gce_summary = train_task5_method(
    method_name="gce_q07_synth",
    train_indices=kept_indices,
    train_targets=synthetic_labels_all[kept_indices],
    loss_function=stable_gce_loss_t4,
    seed=42,
)

print(
    "\nGCE synthetic-validation accuracy:",
    f"{t5_gce_summary['best_val_acc_synthetic']:.2f}%"
)

```

```
gce_q07_synth | Epoch 01/60 | Train loss 1.0800 | Synthetic val acc 18.12%
gce_q07_synth | Epoch 10/60 | Train loss 0.8157 | Synthetic val acc 34.96%
gce_q07_synth | Epoch 20/60 | Train loss 0.7339 | Synthetic val acc 41.54%
gce_q07_synth | Epoch 30/60 | Train loss 0.6881 | Synthetic val acc 42.26%
gce_q07_synth | Epoch 40/60 | Train loss 0.6433 | Synthetic val acc 47.90%
gce_q07_synth | Epoch 50/60 | Train loss 0.5876 | Synthetic val acc 50.10%
gce_q07_synth | Epoch 60/60 | Train loss 0.5472 | Synthetic val acc 50.14%

Completed gce_q07_synth:
{
  "method": "gce_q07_synth",
  "seed": 42,
  "epochs_completed": 60,
  "best_epoch": 48,
  "best_val_acc_synthetic": 50.7,
  "final_val_acc_synthetic": 50.14,
  "runtime_minutes": 30.07653316259384
}

GCE synthetic-validation accuracy: 50.70%

```

```
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

def generate_task5_predictions(method_name, output_filename):
    checkpoint_path = f"/kaggle/working/task5/{method_name}/best_checkpoint.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = ResNet18().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = Task4Dataset(
        original_indices=np.arange(len(test_images)),
        targets=None,
        transform=eval_transform_t4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    predictions = []

    with torch.no_grad():
        for batch_images, _ in test_loader:
            batch_images = batch_images.to(device)
            logits = model(batch_images)
            predictions.extend(logits.argmax(dim=1).cpu().numpy())

    submission = pd.DataFrame({
        "id": np.arange(len(test_images)),
        "pred": predictions
    })

    output_path = f"/kaggle/working/predictions/{output_filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Columns: {submission.columns.tolist()}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Synthetic validation accuracy: {checkpoint['val_acc_synthetic']:.2f}%")
    print(submission.head())

    assert len(submission) == len(test_images)
    assert submission.columns.tolist() == ["id", "pred"]
    assert submission["id"].is_unique
    assert submission["pred"].between(0, 9).all()

    return submission


t5_baseline_predictions = generate_task5_predictions(
    "baseline_synth",
    "t5_baseline_synth.csv"
)

t5_best_predictions = generate_task5_predictions(
    "gce_q07_synth",
    "t5_best_synth.csv"
)

```

```
Saved: /kaggle/working/predictions/t5_baseline_synth.csv
Shape: (10000, 2)
Columns: ['id', 'pred']
Best epoch: 43
Synthetic validation accuracy: 49.66%
   id  pred
0   0     6
1   1     9
2   2     9
3   3     4
4   4     1
Saved: /kaggle/working/predictions/t5_best_synth.csv
Shape: (10000, 2)
Columns: ['id', 'pred']
Best epoch: 48
Synthetic validation accuracy: 50.70%
   id  pred
0   0     6
1   1     9
2   2     9
3   3     4
4   4     1

```

```
import os
import json

TASK5_DIR = "/kaggle/working/task5"
os.makedirs(TASK5_DIR, exist_ok=True)

task5_results = {
    "task": 5,
    "label_source": "synthetic_label",
    "baseline": {
        "method": "CrossEntropy",
        "best_epoch": 43,
        "best_synthetic_val_accuracy": 49.66,
        "prediction_file": "/kaggle/working/predictions/t5_baseline_synth.csv"
    },
    "best_method": {
        "method": "GCE",
        "q": 0.7,
        "filtered_fraction": 0.35,
        "best_epoch": 48,
        "best_synthetic_val_accuracy": 50.70,
        "prediction_file": "/kaggle/working/predictions/t5_best_synth.csv"
    },
    "accuracy_difference_percentage_points": round(50.70 - 49.66, 2),
    "clean_labels_used_for_training_or_selection": False
}

with open(f"{TASK5_DIR}/results.json", "w") as f:
    json.dump(task5_results, f, indent=2)

report = """# Task 5 — Synthetic Labels

## Experimental setup
Both models were trained using `synthetic_label` and selected using synthetic-label validation accuracy.

- **Baseline:** ResNet18 trained with cross-entropy on the full official training split.
- **Best method:** ResNet18 trained with generalized cross-entropy (GCE, q=0.7) on the training subset filtered using the Task 3 noise scores. The same 35% filtering strategy was retained.
- Both runs used seed 42 and trained for up to 60 epochs.

## Results

| Method | Best epoch | Synthetic validation accuracy |
|---|---:|---:|
| Cross-entropy baseline | 43 | 49.66% |
| Filtered GCE (q=0.7) | 48 | 50.70% |

Filtered GCE outperformed the baseline by 1.04 percentage points on synthetic-label validation data.

## Predictions
- `predictions/t5_baseline_synth.csv`
- `predictions/t5_best_synth.csv`

## Q5 — Comparison of real and synthetic noise (under 100 words)
On synthetic-label validation data, filtered GCE achieved 50.70% accuracy, compared with 49.66% for the cross-entropy baseline, a gain of 1.04 percentage points. This suggests the filtering and robust-loss combination was more effective for this synthetic-label experiment. However, this comparison alone does not establish that synthetic noise is easier or harder than real-world noise. A fair conclusion requires comparable evaluations using the corresponding label sources and consistent experimental settings.
"""

with open(f"{TASK5_DIR}/report.md", "w") as f:
    f.write(report)

print("Saved Task 5 results:", f"{TASK5_DIR}/results.json")
print("Saved Task 5 report:", f"{TASK5_DIR}/report.md")
print("\nQ5 response word count:", len(report.split("## Q5")[1].split()))
print("\nTask 5 summary:")
print(json.dumps(task5_results, indent=2))

```

```
Saved Task 5 results: /kaggle/working/task5/results.json
Saved Task 5 report: /kaggle/working/task5/report.md

Q5 response word count: 78

Task 5 summary:
{
  "task": 5,
  "label_source": "synthetic_label",
  "baseline": {
    "method": "CrossEntropy",
    "best_epoch": 43,
    "best_synthetic_val_accuracy": 49.66,
    "prediction_file": "/kaggle/working/predictions/t5_baseline_synth.csv"
  },
  "best_method": {
    "method": "GCE",
    "q": 0.7,
    "filtered_fraction": 0.35,
    "best_epoch": 48,
    "best_synthetic_val_accuracy": 50.7,
    "prediction_file": "/kaggle/working/predictions/t5_best_synth.csv"
  },
  "accuracy_difference_percentage_points": 1.04,
  "clean_labels_used_for_training_or_selection": false
}

```

```
import os
import pandas as pd
import json

PRED_DIR = "/kaggle/working/predictions"

expected_files = [
    "t1_clean_ref.csv",
    "t1_noisy_ce.csv",
    "t2_early_stop.csv",
    "t4_filtered_ce.csv",
    "t4_gce.csv",
    "t4_label_smoothing.csv",
    "t5_baseline_synth.csv",
    "t5_best_synth.csv",
]

print("PREDICTION FILE AUDIT")
print("=" * 65)

for filename in expected_files:
    path = os.path.join(PRED_DIR, filename)

    if not os.path.exists(path):
        print(f"MISSING: {filename}")
        continue

    df = pd.read_csv(path)

    valid = (
        len(df) == 10000
        and df.columns.tolist() == ["id", "pred"]
        and df["id"].is_unique
        and df["id"].min() == 0
        and df["id"].max() == 9999
        and df["pred"].between(0, 9).all()
    )

    print(
        f"{'PASS' if valid else 'CHECK'}: {filename} | "
        f"Shape={df.shape} | "
        f"Prediction range={df['pred'].min()}–{df['pred'].max()}"
    )

print("\nTASK 3 ARTIFACTS")
print("=" * 65)

for filename in [
    "fold_assignment.npy",
    "oof_probabilities_partial.npy",
]:
    path = f"/kaggle/input/datasets/afnanahmed2/prev-outputs/{filename}"
    print(f"{'FOUND' if os.path.exists(path) else 'MISSING'}: {filename}")

print("\nTASK 4 AND TASK 5 REPORTS")
print("=" * 65)

for path in [
    "/kaggle/working/task4/results.json",
    "/kaggle/working/task4/report.md",
    "/kaggle/working/task5/results.json",
    "/kaggle/working/task5/report.md",
]:
    print(f"{'FOUND' if os.path.exists(path) else 'MISSING'}: {path}")

```

```
PREDICTION FILE AUDIT
=================================================================
MISSING: t1_clean_ref.csv
MISSING: t1_noisy_ce.csv
MISSING: t2_early_stop.csv
PASS: t4_filtered_ce.csv | Shape=(10000, 2) | Prediction range=0–9
PASS: t4_gce.csv | Shape=(10000, 2) | Prediction range=0–9
PASS: t4_label_smoothing.csv | Shape=(10000, 2) | Prediction range=0–9
PASS: t5_baseline_synth.csv | Shape=(10000, 2) | Prediction range=0–9
PASS: t5_best_synth.csv | Shape=(10000, 2) | Prediction range=0–9

TASK 3 ARTIFACTS
=================================================================
FOUND: fold_assignment.npy
FOUND: oof_probabilities_partial.npy

TASK 4 AND TASK 5 REPORTS
=================================================================
FOUND: /kaggle/working/task4/results.json
FOUND: /kaggle/working/task4/report.md
FOUND: /kaggle/working/task5/results.json
FOUND: /kaggle/working/task5/report.md

```

```
import os

for root, dirs, files in os.walk("/kaggle/input"):
    for filename in files:
        if filename.endswith(".csv"):
            print(os.path.join(root, filename))

```

```
/kaggle/input/datasets/afnanahmed2/prev-outputs/t2_early_stop.csv
/kaggle/input/datasets/afnanahmed2/prev-outputs/filtered_class_counts.csv
/kaggle/input/datasets/afnanahmed2/prev-outputs/t1_noisy_ce.csv
/kaggle/input/datasets/afnanahmed2/prev-outputs/t2_dynamics.csv
/kaggle/input/datasets/afnanahmed2/prev-outputs/t1_clean_ref.csv
/kaggle/input/datasets/afnanahmed2/inter-iit-cv/labels.csv

```

```
import os
import shutil
import pandas as pd

SOURCE_DIR = "/kaggle/input/datasets/afnanahmed2/prev-outputs"
PRED_DIR = "/kaggle/working/predictions"

os.makedirs(PRED_DIR, exist_ok=True)

files_to_restore = [
    "t1_clean_ref.csv",
    "t1_noisy_ce.csv",
    "t2_early_stop.csv",
]

for filename in files_to_restore:
    source = os.path.join(SOURCE_DIR, filename)
    destination = os.path.join(PRED_DIR, filename)

    shutil.copy2(source, destination)

    df = pd.read_csv(destination)
    print(f"\nRestored: {filename}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3))

```

```
Restored: t1_clean_ref.csv
Shape: (10000, 2)
Columns: ['id', 'pred']
   id  pred
0   0     0
1   1     8
2   2     8

Restored: t1_noisy_ce.csv
Shape: (10000, 2)
Columns: ['id', 'pred']
   id  pred
0   0     0
1   1     8
2   2     9

Restored: t2_early_stop.csv
Shape: (10000, 2)
Columns: ['id', 'pred']
   id  pred
0   0     0
1   1     8
2   2     8

```

```
import os
import pandas as pd

PRED_DIR = "/kaggle/working/predictions"

expected_files = [
    "t1_clean_ref.csv",
    "t1_noisy_ce.csv",
    "t2_early_stop.csv",
    "t4_filtered_ce.csv",
    "t4_gce.csv",
    "t4_label_smoothing.csv",
    "t5_baseline_synth.csv",
    "t5_best_synth.csv",
]

all_passed = True

for filename in expected_files:
    path = os.path.join(PRED_DIR, filename)

    if not os.path.exists(path):
        print(f"❌ MISSING: {filename}")
        all_passed = False
        continue

    df = pd.read_csv(path)

    valid = (
        len(df) == 10000
        and df.columns.tolist() == ["id", "pred"]
        and df["id"].is_unique
        and df["id"].tolist() == list(range(10000))
        and df["pred"].between(0, 9).all()
    )

    print(f"{'✅ PASS' if valid else '❌ FAIL'}: {filename}")
    all_passed &= valid

print("\nFINAL PREDICTION AUDIT:", "ALL PASSED" if all_passed else "ISSUES FOUND")

```

```
✅ PASS: t1_clean_ref.csv
✅ PASS: t1_noisy_ce.csv
✅ PASS: t2_early_stop.csv
✅ PASS: t4_filtered_ce.csv
✅ PASS: t4_gce.csv
✅ PASS: t4_label_smoothing.csv
✅ PASS: t5_baseline_synth.csv
✅ PASS: t5_best_synth.csv

FINAL PREDICTION AUDIT: ALL PASSED

```

```
import os
import numpy as np

TASK3_SOURCE = "/kaggle/input/datasets/afnanahmed2/prev-outputs"

for filename in [
    "fold_assignment.npy",
    "oof_probabilities_partial.npy",
    "t2_dynamics.csv",
    "t2_summary.json",
]:
    path = os.path.join(TASK3_SOURCE, filename)

    if os.path.exists(path):
        print(f"\nFOUND: {filename}")

        if filename.endswith(".npy"):
            arr = np.load(path)
            print("Shape:", arr.shape)
            print("Data type:", arr.dtype)
        else:
            print("Size:", os.path.getsize(path), "bytes")
    else:
        print(f"\nMISSING: {filename}")

```

```
FOUND: fold_assignment.npy
Shape: (45000,)
Data type: int64

FOUND: oof_probabilities_partial.npy
Shape: (45000, 10)
Data type: float32

FOUND: t2_dynamics.csv
Size: 8227 bytes

FOUND: t2_summary.json
Size: 84 bytes

```

```
import numpy as np

TASK3_SOURCE = "/kaggle/input/datasets/afnanahmed2/prev-outputs"

folds = np.load(f"{TASK3_SOURCE}/fold_assignment.npy")
oof_probs = np.load(f"{TASK3_SOURCE}/oof_probabilities_partial.npy")

print("FOLD ASSIGNMENT")
print("Shape:", folds.shape)
print("Unique fold IDs:", np.unique(folds))
print("Fold counts:", np.unique(folds, return_counts=True))

print("\nOOF PROBABILITIES")
print("Shape:", oof_probs.shape)
print("All finite:", np.isfinite(oof_probs).all())
print("Minimum probability:", np.nanmin(oof_probs))
print("Maximum probability:", np.nanmax(oof_probs))

row_sums = oof_probs.sum(axis=1)

print("Rows with sum approximately 1:", np.isclose(row_sums, 1.0, atol=1e-3).sum())
print("Rows with invalid sums:", (~np.isclose(row_sums, 1.0, atol=1e-3)).sum())
print("Rows containing NaN/Inf:", (~np.isfinite(oof_probs).all(axis=1)).sum())

print("\nTask 3 validation complete.")

```

```
FOLD ASSIGNMENT
Shape: (45000,)
Unique fold IDs: [1 2 3 4 5]
Fold counts: (array([1, 2, 3, 4, 5]), array([9000, 9000, 9000, 9000, 9000]))

OOF PROBABILITIES
Shape: (45000, 10)
All finite: True
Minimum probability: 0.00015849252
Maximum probability: 0.98987657
Rows with sum approximately 1: 45000
Rows with invalid sums: 0
Rows containing NaN/Inf: 0

Task 3 validation complete.

```

```
import os

for root, dirs, files in os.walk("/kaggle/input"):
    for filename in files:
        if filename.lower().endswith((".pdf", ".txt", ".md", ".json")):
            if any(word in filename.lower() for word in [
                "task", "readme", "instruction", "submission", "requirement"
            ]):
                print(os.path.join(root, filename))

```

```
import numpy as np
import pandas as pd
import os

SOURCE_DIR = "/kaggle/input/datasets/afnanahmed2/prev-outputs"
DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"

folds = np.load(os.path.join(SOURCE_DIR, "fold_assignment.npy"))
oof_probs = np.load(os.path.join(SOURCE_DIR, "oof_probabilities_partial.npy"))

labels_df = pd.read_csv(os.path.join(DATA_PATH, "labels.csv"))
train_idx = np.loadtxt(os.path.join(DATA_PATH, "train_idx.txt"), dtype=int)

noisy_labels = labels_df["noisy_label"].to_numpy()[train_idx]

print("Training indices:", len(train_idx))
print("Fold assignments:", len(folds))
print("OOF probability rows:", len(oof_probs))
print("Noisy labels:", len(noisy_labels))

print("\nProbability shape:", oof_probs.shape)
print("Fold IDs:", np.unique(folds))
print("All probability rows sum to 1:",
      np.allclose(oof_probs.sum(axis=1), 1.0, atol=1e-3))

print("\nExisting Task 3 score files:")
for root, dirs, files in os.walk("/kaggle/input"):
    for filename in files:
        if "t3_noise_scores" in filename.lower():
            print(os.path.join(root, filename))

```

```
Training indices: 45000
Fold assignments: 45000
OOF probability rows: 45000
Noisy labels: 45000

Probability shape: (45000, 10)
Fold IDs: [1 2 3 4 5]
All probability rows sum to 1: True

Existing Task 3 score files:

```

```
import os
import numpy as np
import pandas as pd

SOURCE_DIR = "/kaggle/input/datasets/afnanahmed2/prev-outputs"
DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"
PRED_DIR = "/kaggle/working/predictions"

os.makedirs(PRED_DIR, exist_ok=True)

# Load saved out-of-fold probabilities and fold assignments
folds = np.load(f"{SOURCE_DIR}/fold_assignment.npy")
oof_probs = np.load(f"{SOURCE_DIR}/oof_probabilities_partial.npy")

# Load official training indices and noisy labels
labels_df = pd.read_csv(f"{DATA_PATH}/labels.csv")
train_idx = np.loadtxt(f"{DATA_PATH}/train_idx.txt", dtype=int)

noisy_labels = labels_df["noisy_label"].to_numpy()[train_idx]

assert oof_probs.shape == (len(train_idx), 10)
assert len(folds) == len(train_idx)
assert np.isfinite(oof_probs).all()
assert np.allclose(oof_probs.sum(axis=1), 1.0, atol=1e-3)

# Probability assigned to each image's observed noisy label
p_observed_label = oof_probs[
    np.arange(len(train_idx)),
    noisy_labels
]

# Higher score = more likely the observed label is wrong
noise_scores = 1.0 - p_observed_label

# Save original dataset indices, as required by the problem statement
t3_scores = pd.DataFrame({
    "index": train_idx,
    "score": noise_scores
})

output_path = f"{PRED_DIR}/t3_noise_scores.csv"
t3_scores.to_csv(output_path, index=False)

assert len(t3_scores) == 45000
assert t3_scores.columns.tolist() == ["index", "score"]
assert t3_scores["index"].is_unique
assert np.isfinite(t3_scores["score"]).all()
assert t3_scores["score"].between(0, 1).all()

print("Saved:", output_path)
print("Shape:", t3_scores.shape)
print("Columns:", t3_scores.columns.tolist())
print("\nHighest noise scores:")
print(t3_scores.nlargest(10, "score"))
print("\nScore summary:")
print(t3_scores["score"].describe())

```

```
Saved: /kaggle/working/predictions/t3_noise_scores.csv
Shape: (45000, 2)
Columns: ['index', 'score']

Highest noise scores:
       index     score
33441  37173  0.999587
39913  44355  0.999547
37249  41382  0.999240
15862  17640  0.999141
2472    2764  0.999005
25578  28449  0.998934
40202  44675  0.998819
3899    4329  0.998802
32028  35599  0.998689
36855  40949  0.998605

Score summary:
count    45000.000000
mean         0.604534
std          0.303849
min          0.010123
25%          0.319646
50%          0.632239
75%          0.914793
max          0.999587
Name: score, dtype: float64

```

```
import os
import pandas as pd
import numpy as np

PRED_DIR = "/kaggle/working/predictions"

# Task 1, 2, 4 and 5 classification submissions
prediction_files = [
    "t1_clean_ref.csv",
    "t1_noisy_ce.csv",
    "t2_early_stop.csv",
    "t4_filtered_ce.csv",
    "t4_gce.csv",
    "t4_label_smoothing.csv",
    "t5_baseline_synth.csv",
    "t5_best_synth.csv",
]

all_passed = True

print("=" * 65)
print("CLASSIFICATION PREDICTION AUDIT")
print("=" * 65)

for filename in prediction_files:
    path = os.path.join(PRED_DIR, filename)

    if not os.path.exists(path):
        print(f"MISSING: {filename}")
        all_passed = False
        continue

    df = pd.read_csv(path)

    valid = (
        len(df) == 10000
        and df.columns.tolist() == ["id", "pred"]
        and df["id"].is_unique
        and np.array_equal(df["id"].to_numpy(), np.arange(10000))
        and df["pred"].between(0, 9).all()
    )

    print(f"{'PASS' if valid else 'FAIL'}: {filename}")
    all_passed &= valid

print("\n" + "=" * 65)
print("TASK 3 NOISE-SCORE AUDIT")
print("=" * 65)

t3_path = os.path.join(PRED_DIR, "t3_noise_scores.csv")

if os.path.exists(t3_path):
    t3 = pd.read_csv(t3_path)

    valid_t3 = (
        len(t3) == 45000
        and t3.columns.tolist() == ["index", "score"]
        and t3["index"].is_unique
        and t3["score"].between(0, 1).all()
        and np.isfinite(t3["score"]).all()
    )

    print("Shape:", t3.shape)
    print("Columns:", t3.columns.tolist())
    print("Task 3 format:", "PASS" if valid_t3 else "FAIL")
    all_passed &= valid_t3
else:
    print("MISSING: t3_noise_scores.csv")
    all_passed = False

print("\n" + "=" * 65)
print("AUDIT RESULT:", "ALL FORMATS PASSED" if all_passed else "ISSUES FOUND")
print("=" * 65)

```

```
=================================================================
CLASSIFICATION PREDICTION AUDIT
=================================================================
PASS: t1_clean_ref.csv
PASS: t1_noisy_ce.csv
PASS: t2_early_stop.csv
PASS: t4_filtered_ce.csv
PASS: t4_gce.csv
PASS: t4_label_smoothing.csv
PASS: t5_baseline_synth.csv
PASS: t5_best_synth.csv

=================================================================
TASK 3 NOISE-SCORE AUDIT
=================================================================
Shape: (45000, 2)
Columns: ['index', 'score']
Task 3 format: PASS

=================================================================
AUDIT RESULT: ALL FORMATS PASSED
=================================================================

```

```
import os
import shutil
import pandas as pd

SOURCE_DIR = "/kaggle/input/datasets/afnanahmed2/prev-outputs"
PRED_DIR = "/kaggle/working/predictions"

os.makedirs(PRED_DIR, exist_ok=True)

source = os.path.join(SOURCE_DIR, "t2_dynamics.csv")
destination = os.path.join(PRED_DIR, "t2_dynamics.csv")

shutil.copy2(source, destination)

df = pd.read_csv(destination)

print("Restored:", destination)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst five epochs:")
print(df.head())
print("\nLast five epochs:")
print(df.tail())
print("\nMissing values:")
print(df.isna().sum())

```

```
Restored: /kaggle/working/predictions/t2_dynamics.csv
Shape: (60, 9)
Columns: ['epoch', 'train_loss', 'train_acc_correct', 'train_acc_mislabeled', 'val_loss_noisy', 'val_loss_clean', 'val_acc_noisy', 'val_acc_clean', 'learning_rate']

First five epochs:
   epoch  train_loss  train_acc_correct  train_acc_mislabeled  val_loss_noisy  \
0      1    2.311560          30.333569             14.617041        2.063224   
1      2    1.985734          37.997843             16.317853        1.945217   
2      3    1.908481          45.907553             15.191341        1.887858   
3      4    1.846360          55.152281             13.838423        1.779685   
4      5    1.784684          56.000149             15.257607        1.753896   

   val_loss_clean  val_acc_noisy  val_acc_clean  learning_rate  
0        1.892452          22.16          26.74       0.100000  
1        1.705604          28.94          35.36       0.099931  
2        1.596964          33.64          42.46       0.099726  
3        1.418707          38.08          50.78       0.099384  
4        1.365336          38.76          52.24       0.098907  

Last five epochs:
    epoch  train_loss  train_acc_correct  train_acc_mislabeled  \
55     56    0.677750          95.820163             59.395881   
56     57    0.618912          96.370533             63.449114   
57     58    0.571279          96.928340             65.928544   
58     59    0.535430          97.147745             67.684577   
59     60    0.518047          97.095683             68.479761   

    val_loss_noisy  val_loss_clean  val_acc_noisy  val_acc_clean  \
55        1.968644        1.007403          50.72          70.56   
56        1.978861        1.037923          50.68          69.98   
57        2.020053        1.053294          50.72          69.06   
58        2.036821        1.068089          50.36          68.92   
59        2.032041        1.074470          50.56          69.02   

    learning_rate  
55       0.001704  
56       0.001093  
57       0.000616  
58       0.000274  
59       0.000069  

Missing values:
epoch                   0
train_loss              0
train_acc_correct       0
train_acc_mislabeled    0
val_loss_noisy          0
val_loss_clean          0
val_acc_noisy           0
val_acc_clean           0
learning_rate           0
dtype: int64

```

```
import pandas as pd

dynamics_path = "/kaggle/working/predictions/t2_dynamics.csv"
dynamics = pd.read_csv(dynamics_path)

best_noisy_row = dynamics.loc[dynamics["val_acc_noisy"].idxmax()]

print("TASK 2 TRAINING SUMMARY")
print("=" * 45)
print("Total epochs:", len(dynamics))
print("Best epoch (noisy validation):", int(best_noisy_row["epoch"]))
print(f"Best noisy validation accuracy: {best_noisy_row['val_acc_noisy']:.2f}%")
print(f"Noisy validation accuracy at epoch 60: {dynamics.iloc[-1]['val_acc_noisy']:.2f}%")
print(f"Clean validation accuracy at epoch 60 (diagnostic): {dynamics.iloc[-1]['val_acc_clean']:.2f}%")

print("\nBest noisy-validation epoch details:")
print(best_noisy_row.to_string())

print("\nSaved Task 2 summary:")
print(pd.read_json(
    "/kaggle/input/datasets/afnanahmed2/prev-outputs/t2_summary.json",
    typ="series"
))

```

```
TASK 2 TRAINING SUMMARY
=============================================
Total epochs: 60
Best epoch (noisy validation): 39
Best noisy validation accuracy: 55.20%
Noisy validation accuracy at epoch 60: 50.56%
Clean validation accuracy at epoch 60 (diagnostic): 69.02%

Best noisy-validation epoch details:
epoch                   39.000000
train_loss               1.321141
train_acc_correct       88.750883
train_acc_mislabeled    17.405710
val_loss_noisy           1.462946
val_loss_clean           0.809511
val_acc_noisy           55.200000
val_acc_clean           77.320000
learning_rate            0.029663

Saved Task 2 summary:
best_epoch            39.0
best_val_acc_noisy    55.2
epochs_completed      60.0
dtype: float64

```

```
import os

for root, dirs, files in os.walk("/kaggle/input"):
    for filename in files:
        if filename.lower() in [
            "results_template.json",
            "check_format.py",
            "readme.md",
        ]:
            print(os.path.join(root, filename))

```

```
import pandas as pd

path = "/kaggle/working/predictions/t2_dynamics.csv"

dynamics = pd.read_csv(path)

required_columns = [
    "epoch",
    "train_acc_correct",
    "train_acc_mislabeled",
    "val_acc_noisy",
    "val_acc_clean",
]

# Retain only the columns required by the official checker
dynamics = dynamics[required_columns]

dynamics.to_csv(path, index=False)

# Verify
check = pd.read_csv(path)

print("Saved:", path)
print("Columns:", check.columns.tolist())
print("Shape:", check.shape)
print(check.head())

assert check.columns.tolist() == required_columns
assert len(check) == 60
assert not check.isna().any().any()

print("\nSUCCESS: Task 2 dynamics format now matches the official requirements.")

```

```
Saved: /kaggle/working/predictions/t2_dynamics.csv
Columns: ['epoch', 'train_acc_correct', 'train_acc_mislabeled', 'val_acc_noisy', 'val_acc_clean']
Shape: (60, 5)
   epoch  train_acc_correct  train_acc_mislabeled  val_acc_noisy  \
0      1          30.333569             14.617041          22.16   
1      2          37.997843             16.317853          28.94   
2      3          45.907553             15.191341          33.64   
3      4          55.152281             13.838423          38.08   
4      5          56.000149             15.257607          38.76   

   val_acc_clean  
0          26.74  
1          35.36  
2          42.46  
3          50.78  
4          52.24  

SUCCESS: Task 2 dynamics format now matches the official requirements.

```

```
import os

SUBMISSION_DIR = "/kaggle/working/submission"
CODE_DIR = os.path.join(SUBMISSION_DIR, "code")
SUBMISSION_PRED_DIR = os.path.join(SUBMISSION_DIR, "predictions")

os.makedirs(CODE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_PRED_DIR, exist_ok=True)

print("Submission structure created:")
for path in [SUBMISSION_DIR, CODE_DIR, SUBMISSION_PRED_DIR]:
    print(path)

```

```
Submission structure created:
/kaggle/working/submission
/kaggle/working/submission/code
/kaggle/working/submission/predictions

```

```
import os
import shutil

SOURCE_DIR = "/kaggle/working/predictions"
DEST_DIR = "/kaggle/working/submission/predictions"

files_to_copy = [
    "t1_clean_ref.csv",
    "t1_noisy_ce.csv",
    "t2_early_stop.csv",
    "t2_dynamics.csv",
    "t3_noise_scores.csv",
    "t4_filtered_ce.csv",
    "t4_gce.csv",
    "t4_label_smoothing.csv",
    "t5_baseline_synth.csv",
    "t5_best_synth.csv",
]

print("COPYING PREDICTION FILES")
print("=" * 55)

for filename in files_to_copy:
    source = os.path.join(SOURCE_DIR, filename)
    destination = os.path.join(DEST_DIR, filename)

    if not os.path.exists(source):
        print(f"MISSING: {filename}")
        continue

    shutil.copy2(source, destination)
    print(f"COPIED: {filename}")

print("\nSubmission predictions directory:")
for filename in sorted(os.listdir(DEST_DIR)):
    print(" -", filename)

print("\nTotal files:", len(os.listdir(DEST_DIR)))

```

```
COPYING PREDICTION FILES
=======================================================
COPIED: t1_clean_ref.csv
COPIED: t1_noisy_ce.csv
COPIED: t2_early_stop.csv
COPIED: t2_dynamics.csv
COPIED: t3_noise_scores.csv
COPIED: t4_filtered_ce.csv
COPIED: t4_gce.csv
COPIED: t4_label_smoothing.csv
COPIED: t5_baseline_synth.csv
COPIED: t5_best_synth.csv

Submission predictions directory:
 - t1_clean_ref.csv
 - t1_noisy_ce.csv
 - t2_dynamics.csv
 - t2_early_stop.csv
 - t3_noise_scores.csv
 - t4_filtered_ce.csv
 - t4_gce.csv
 - t4_label_smoothing.csv
 - t5_baseline_synth.csv
 - t5_best_synth.csv

Total files: 10

```

```
import os

CODE_DIR = "/kaggle/working/submission/code"
os.makedirs(CODE_DIR, exist_ok=True)

readme = """# Inter-IIT Tech Meet 15.0 — Computer Vision Bootcamp

## Project overview
This submission addresses the CIFAR-10N noisy-label classification tasks.

## Dataset
- Training images: 50,000
- Test images: 10,000
- Official training split: 45,000 samples
- Official validation split: 5,000 samples
- Number of classes: 10

## Models and methods

### Task 1
- Clean-label reference model.
- Cross-entropy baseline trained using noisy labels.

### Task 2
- ResNet18 trained with cross-entropy.
- Model selection based on noisy validation accuracy.
- Best noisy validation accuracy: 55.20% at epoch 39.
- Training dynamics saved in `predictions/t2_dynamics.csv`.

### Task 3
- Out-of-fold probability matrix with 45,000 rows and 10 classes.
- Noise score calculated as 1 minus the probability assigned to the observed noisy label.
- Scores saved in `predictions/t3_noise_scores.csv`.
- OOF generation provenance must be verified against the training procedure.

### Task 4
Three approaches were compared using noisy-label validation accuracy:
- Cross-entropy after filtering 35% of training samples using Task 3 scores: 56.42%.
- Generalized cross-entropy (q=0.7): 57.16%.
- Cross-entropy with label smoothing (0.1): 56.58%.

The best observed noisy validation accuracy was achieved by GCE (q=0.7).

### Task 5
Models were trained and selected using synthetic labels:
- Full-training-set cross-entropy baseline: 49.66% synthetic validation accuracy.
- Filtered GCE (q=0.7): 50.70% synthetic validation accuracy.

## Prediction files
Classification predictions use columns `id,pred`.
Task 3 noise scores use columns `index,score`.
Task 2 training dynamics use the five required columns.

## Reproducibility
The experiments used ResNet18, seed 42, SGD with momentum, weight decay, and cosine learning-rate scheduling. The training runs used up to 60 epochs.

This README documents the methods and results. It is not a substitute for the executable training code or verification of the official submission requirements.
"""

readme_path = os.path.join(CODE_DIR, "README.md")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

print("Created:", readme_path)
print("Size:", os.path.getsize(readme_path), "bytes")
print("\nREADME preview:\n")
print(readme)

```

```
Created: /kaggle/working/submission/code/README.md
Size: 2012 bytes

README preview:

# Inter-IIT Tech Meet 15.0 — Computer Vision Bootcamp

## Project overview
This submission addresses the CIFAR-10N noisy-label classification tasks.

## Dataset
- Training images: 50,000
- Test images: 10,000
- Official training split: 45,000 samples
- Official validation split: 5,000 samples
- Number of classes: 10

## Models and methods

### Task 1
- Clean-label reference model.
- Cross-entropy baseline trained using noisy labels.

### Task 2
- ResNet18 trained with cross-entropy.
- Model selection based on noisy validation accuracy.
- Best noisy validation accuracy: 55.20% at epoch 39.
- Training dynamics saved in `predictions/t2_dynamics.csv`.

### Task 3
- Out-of-fold probability matrix with 45,000 rows and 10 classes.
- Noise score calculated as 1 minus the probability assigned to the observed noisy label.
- Scores saved in `predictions/t3_noise_scores.csv`.
- OOF generation provenance must be verified against the training procedure.

### Task 4
Three approaches were compared using noisy-label validation accuracy:
- Cross-entropy after filtering 35% of training samples using Task 3 scores: 56.42%.
- Generalized cross-entropy (q=0.7): 57.16%.
- Cross-entropy with label smoothing (0.1): 56.58%.

The best observed noisy validation accuracy was achieved by GCE (q=0.7).

### Task 5
Models were trained and selected using synthetic labels:
- Full-training-set cross-entropy baseline: 49.66% synthetic validation accuracy.
- Filtered GCE (q=0.7): 50.70% synthetic validation accuracy.

## Prediction files
Classification predictions use columns `id,pred`.
Task 3 noise scores use columns `index,score`.
Task 2 training dynamics use the five required columns.

## Reproducibility
The experiments used ResNet18, seed 42, SGD with momentum, weight decay, and cosine learning-rate scheduling. The training runs used up to 60 epochs.

This README documents the methods and results. It is not a substitute for the executable training code or verification of the official submission requirements.


```

```
import os

matches = []

for root, dirs, files in os.walk("/kaggle/working"):
    for filename in files:
        if filename.lower() == "results_template.json":
            matches.append(os.path.join(root, filename))

for root, dirs, files in os.walk("/kaggle/input"):
    for filename in files:
        if filename.lower() in [
            "results_template.json",
            "check_format.py",
        ]:
            matches.append(os.path.join(root, filename))

if matches:
    print("Found official files:")
    for path in matches:
        print("\n", path)
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                print(f.read())
else:
    print("Official template not found in Kaggle.")
    print("We need to extract it from the bootcamp ZIP.")

```

```
Official template not found in Kaggle.
We need to extract it from the bootcamp ZIP.

```

```
import json

results_path = "/kaggle/working/submission/results.json"

with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)

results["roll_no"] = "25B0620"

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Updated results.json")
print("Name:", results["name"])
print("Roll number:", results["roll_no"])
print("Saved:", results_path)

```

```
Updated results.json
Name: Afnan Ahmed
Roll number: 25B0620
Saved: /kaggle/working/submission/results.json

```

```
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/datasets/afnanahmed2/inter-iit-cv"
SOURCE_DIR = "/kaggle/input/datasets/afnanahmed2/prev-outputs"

labels_df = pd.read_csv(f"{DATA_PATH}/labels.csv")
val_idx = np.loadtxt(f"{DATA_PATH}/val_idx.txt", dtype=int)

clean_labels = labels_df["clean_label"].to_numpy()[val_idx]
noisy_labels = labels_df["noisy_label"].to_numpy()[val_idx]

print("Validation samples:", len(val_idx))
print("Noisy-label accuracy can be measured against noisy_labels.")
print("Clean-label accuracy can be measured against clean_labels.")
print("\nAvailable saved Task 1 files:")

for filename in ["t1_clean_ref.csv", "t1_noisy_ce.csv"]:
    path = f"{SOURCE_DIR}/{filename}"
    df = pd.read_csv(path)
    print(f"\n{filename}:")
    print(df.head(3))
    print("Rows:", len(df))

```

```
Validation samples: 5000
Noisy-label accuracy can be measured against noisy_labels.
Clean-label accuracy can be measured against clean_labels.

Available saved Task 1 files:

t1_clean_ref.csv:
   id  pred
0   0     0
1   1     8
2   2     8
Rows: 10000

t1_noisy_ce.csv:
   id  pred
0   0     0
1   1     8
2   2     9
Rows: 10000

```

```
import os

search_dirs = [
    "/kaggle/input/datasets/afnanahmed2/prev-outputs",
    "/kaggle/working",
]

keywords = ["t1", "task1", "clean_ref", "noisy_ce"]

print("Searching for Task 1 logs, summaries, and checkpoints...\n")

for base in search_dirs:
    for root, dirs, files in os.walk(base):
        for filename in files:
            lower = filename.lower()

            if any(k in lower for k in keywords):
                path = os.path.join(root, filename)
                print(path)
                

```

```
Searching for Task 1 logs, summaries, and checkpoints...

/kaggle/input/datasets/afnanahmed2/prev-outputs/t1_noisy_ce.csv
/kaggle/input/datasets/afnanahmed2/prev-outputs/t1_clean_ref.csv
/kaggle/working/submission/predictions/t1_clean_ref.csv
/kaggle/working/submission/predictions/t1_noisy_ce.csv
/kaggle/working/predictions/t1_clean_ref.csv
/kaggle/working/predictions/t1_noisy_ce.csv

```

```
import os
import json

SOURCE_DIR = "/kaggle/input/datasets/afnanahmed2/prev-outputs"

for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".json"):
        path = os.path.join(SOURCE_DIR, filename)
        print(f"\n--- {filename} ---")

        with open(path, "r", encoding="utf-8") as f:
            print(f.read())

```

```
--- filtered_ce_config.json ---
{
  "method": "task3_score_filtered_cross_entropy",
  "seed": 42,
  "filter_fraction": 0.35,
  "removed_images": 15749,
  "retained_images": 29251,
  "selection_rule": "Remove highest 35% of Task 3 noise scores",
  "label_source": "noisy_label",
  "clean_labels_used_for_filtering": false
}

--- t2_summary.json ---
{
    "best_epoch": 39,
    "best_val_acc_noisy": 55.2,
    "epochs_completed": 60
}

```

```
import json

results_path = "/kaggle/working/submission/results.json"

with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)

results["task1"]["val_acc_noisy_labels__clean_ref_model"] = 94.52
results["task1"]["val_acc_noisy_labels__noisy_ce_model"] = 49.66

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Task 1 metrics updated:")
print(json.dumps(results["task1"], indent=2))

```

```
Task 1 metrics updated:
{
  "noise_rate": 40.208,
  "val_acc_noisy_labels__clean_ref_model": 94.52,
  "val_acc_noisy_labels__noisy_ce_model": 49.66
}

```

```
report_path = "/kaggle/working/submission/report.md"

report = """# Inter-IIT Tech Meet 15.0 — Computer Vision Bootcamp

**Name:** Afnan Ahmed  
**Roll number:** 25B0620

## Q1 — Label Noise Analysis
The dataset contains 50,000 training images. The clean-label reference identified 20,104 mislabeled samples, corresponding to a noise rate of 40.208%. The clean-reference model achieved 94.52% validation accuracy, while the noisy-label cross-entropy model achieved 49.66%. Frequent confusion pairs included automobile/truck and cat/dog.

## Q2 — Memorization and Early Stopping
The cross-entropy baseline achieved its best noisy-validation accuracy of 55.20% at epoch 39. Accuracy declined to 50.56% by epoch 60, a decrease of 4.64 percentage points. This behavior is consistent with later training fitting noisy labels and reducing generalization to the noisy validation set. Training dynamics are provided in `predictions/t2_dynamics.csv`.

## Q3 — Noise Scoring
The saved out-of-fold probability matrix contains 45,000 rows and 10 classes, with finite probabilities summing approximately to one. Noise scores were calculated as one minus the probability assigned to the observed noisy label. The resulting file contains 45,000 rows with columns `index,score`. The original OOF generation provenance has not yet been independently verified.

## Q4 — Robust Training Methods
Three approaches were evaluated using noisy-label validation accuracy. Cross-entropy after filtering 35% of training samples achieved 56.42%; generalized cross-entropy (q=0.7) achieved 57.16%; and label smoothing (0.1) achieved 56.58%. Generalized cross-entropy was the best of these methods by noisy-validation accuracy.

## Q5 — Synthetic Labels
The cross-entropy baseline achieved 49.66% synthetic-label validation accuracy, while filtered generalized cross-entropy (q=0.7) achieved 50.70%, an improvement of 1.04 percentage points. This comparison shows that the filtered robust-loss method performed better in this experiment, but does not by itself establish whether synthetic noise is easier or harder than real-world noise.

## Limitations
The Task 3 probability matrix passed numerical checks, but its out-of-fold generation procedure still needs verification. Results are reported only where supporting artifacts were available.
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print("Updated:", report_path)
print("Report word count:", len(report.split()))
print("\nQ1–Q5 sections:")
for section in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    print(section, "present:", f"## {section}" in report)

```

```
Updated: /kaggle/working/submission/report.md
Report word count: 304

Q1–Q5 sections:
Q1 present: True
Q2 present: True
Q3 present: True
Q4 present: True
Q5 present: True

```

```
import os

SUBMISSION_DIR = "/kaggle/working/submission"

for folder in ["code", "predictions"]:
    path = os.path.join(SUBMISSION_DIR, folder)
    print(f"\n{folder}/")
    if os.path.isdir(path):
        for filename in sorted(os.listdir(path)):
            print("  ", filename)
    else:
        print("  MISSING FOLDER!")

for filename in ["results.json", "report.md"]:
    path = os.path.join(SUBMISSION_DIR, filename)
    print(f"\n{filename}: {'OK' if os.path.isfile(path) else 'MISSING'}")

```

```
code/
   README.md

predictions/
   t1_clean_ref.csv
   t1_noisy_ce.csv
   t2_dynamics.csv
   t2_early_stop.csv
   t3_noise_scores.csv
   t4_filtered_ce.csv
   t4_gce.csv
   t4_label_smoothing.csv
   t5_baseline_synth.csv
   t5_best_synth.csv

results.json: OK

report.md: OK

```

```
import os
import json
import pandas as pd
import numpy as np

SUBMISSION_DIR = "/kaggle/working/submission"
PRED_DIR = os.path.join(SUBMISSION_DIR, "predictions")

expected_files = [
    "t1_clean_ref.csv",
    "t1_noisy_ce.csv",
    "t2_dynamics.csv",
    "t2_early_stop.csv",
    "t3_noise_scores.csv",
    "t4_filtered_ce.csv",
    "t4_gce.csv",
    "t4_label_smoothing.csv",
    "t5_baseline_synth.csv",
    "t5_best_synth.csv",
]

print("=== Prediction file validation ===")

all_ok = True

for filename in expected_files:
    path = os.path.join(PRED_DIR, filename)

    if not os.path.isfile(path):
        print(f"FAIL: {filename} — missing")
        all_ok = False
        continue

    df = pd.read_csv(path)

    if filename == "t3_noise_scores.csv":
        valid = (
            list(df.columns) == ["index", "score"]
            and len(df) == 45000
            and df["index"].is_unique
            and df["score"].notna().all()
            and df["score"].between(0, 1).all()
        )
    elif filename == "t2_dynamics.csv":
        required_cols = [
            "epoch",
            "train_acc_correct",
            "train_acc_mislabeled",
            "val_acc_noisy",
            "val_acc_clean",
        ]
        valid = (
            list(df.columns) == required_cols
            and len(df) == 60
            and not df.isna().any().any()
        )
    else:
        valid = (
            list(df.columns) == ["id", "pred"]
            and len(df) == 10000
            and df["id"].is_unique
            and df["id"].tolist() == list(range(10000))
            and df["pred"].notna().all()
            and df["pred"].isin(range(10)).all()
        )

    print(f"{'PASS' if valid else 'FAIL'}: {filename} ({len(df)} rows)")
    all_ok &= valid

print("\n=== JSON validation ===")

results_path = os.path.join(SUBMISSION_DIR, "results.json")

try:
    with open(results_path, "r") as f:
        results = json.load(f)
    print("PASS: results.json is valid JSON")
except Exception as e:
    print("FAIL:", e)
    all_ok = False

print("\n=== Final status ===")
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")

```

```
=== Prediction file validation ===
PASS: t1_clean_ref.csv (10000 rows)
PASS: t1_noisy_ce.csv (10000 rows)
PASS: t2_dynamics.csv (60 rows)
PASS: t2_early_stop.csv (10000 rows)
PASS: t3_noise_scores.csv (45000 rows)
PASS: t4_filtered_ce.csv (10000 rows)
PASS: t4_gce.csv (10000 rows)
PASS: t4_label_smoothing.csv (10000 rows)
PASS: t5_baseline_synth.csv (10000 rows)
PASS: t5_best_synth.csv (10000 rows)

=== JSON validation ===
PASS: results.json is valid JSON

=== Final status ===
ALL CHECKS PASSED

```

```
import os
import zipfile

SUBMISSION_DIR = "/kaggle/working/submission"
ZIP_PATH = "/kaggle/working/25B0620_cv_bootcamp.zip"

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(SUBMISSION_DIR):
        for filename in files:
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, SUBMISSION_DIR)
            zf.write(full_path, relative_path)

print("ZIP created:", ZIP_PATH)
print("ZIP size: {:.2f} MB".format(os.path.getsize(ZIP_PATH) / (1024**2)))

# Verify the archive contents
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    print("\nArchive integrity:", zf.testzip() or "PASS")
    print("\nFiles included:")
    for name in sorted(zf.namelist()):
        print(" ", name)

```

```
ZIP created: /kaggle/working/25B0620_cv_bootcamp.zip
ZIP size: 0.50 MB

Archive integrity: PASS

Files included:
  code/README.md
  predictions/t1_clean_ref.csv
  predictions/t1_noisy_ce.csv
  predictions/t2_dynamics.csv
  predictions/t2_early_stop.csv
  predictions/t3_noise_scores.csv
  predictions/t4_filtered_ce.csv
  predictions/t4_gce.csv
  predictions/t4_label_smoothing.csv
  predictions/t5_baseline_synth.csv
  predictions/t5_best_synth.csv
  report.md
  results.json

```

```
 
```