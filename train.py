import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from models.unet import UNet
from models.classifier import TumorClassifier

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 256
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 1e-4

# ── Dataset ────────────────────────────────────────────────────────────────────
class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def load_dataset():
    paths, labels = [], []
    for label, folder in enumerate(["no", "yes"]):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"  ⚠️  Folder not found: {folder_path}")
            continue
        for f in os.listdir(folder_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder_path, f))
                labels.append(label)
    print(f"  📊 Total images: {len(paths)} (0=no tumor, 1=tumor)")
    return paths, labels

# ── Train Classifier ────────────────────────────────────────────────────────────
def train_classifier():
    print("\n🧠 Training Classifier (ResNet18)...")
    paths, labels = load_dataset()
    if len(paths) == 0:
        print("  ❌ No data found. Run download_dataset.py first.")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=0.2, stratify=labels)
    train_ds = BrainMRIDataset(X_train, y_train, transform)
    val_ds   = BrainMRIDataset(X_val,   y_val,   transform)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = TumorClassifier(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
        acc = correct / len(val_ds) * 100
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "classifier.pth"))
            print(f"  💾 Saved best classifier ({acc:.1f}%)")

    print(f"\n✅ Classifier training done. Best accuracy: {best_acc:.1f}%")

# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_classifier()
    print("\n✅ All training complete!")
    print("   Saved: models/classifier.pth")
    print("   Next step: run app.py")
