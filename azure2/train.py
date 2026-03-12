import argparse
import os
import random
from collections import Counter
import inspect

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from dataset import OrientationDataset
from model import OrientationNet


def _list_images(data_dir):
    paths = []
    for file in os.listdir(data_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(data_dir, file))
    return paths


def _split_paths(paths, val_split, seed):
    paths = list(paths)
    random.Random(seed).shuffle(paths)

    if val_split <= 0:
        return paths, []

    if len(paths) < 2:
        return paths, []

    val_count = max(1, int(round(len(paths) * val_split)))
    val_count = min(val_count, len(paths) - 1)

    val_paths = paths[:val_count]
    train_paths = paths[val_count:]
    return train_paths, val_paths


def _make_criterion(label_smoothing):
    ce_init = inspect.signature(nn.CrossEntropyLoss.__init__).parameters
    if "label_smoothing" in ce_init:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing and label_smoothing > 0:
        print("Warning: torch CrossEntropyLoss does not support label_smoothing in this version.")
    return nn.CrossEntropyLoss()


def _unfreeze_last_n_blocks(model, n):
    if n <= 0:
        return

    features = model.model.features
    blocks = list(features.children())

    # freeze all backbone blocks, then unfreeze last N blocks
    for p in features.parameters():
        p.requires_grad = False

    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True


def _evaluate(model, dataloader, device):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1).cpu().tolist()
            preds.extend(pred)
            labels.extend(target.tolist())

    if not labels:
        return 0.0, None

    correct = sum(int(p == y) for p, y in zip(preds, labels))
    acc = correct / len(labels)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    return acc, cm


def train(data_dir, model_dir, *, epochs, batch_size, lr, weight_decay, val_split, seed, img_size, label_smoothing, unfreeze_last_n, patience):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    paths = _list_images(data_dir)
    if not paths:
        raise RuntimeError(f"No images found in {data_dir}")

    train_paths, val_paths = _split_paths(paths, val_split=val_split, seed=seed)

    train_dataset = OrientationDataset(paths=train_paths, img_size=img_size, train=True)
    val_dataset = OrientationDataset(paths=val_paths, img_size=img_size, train=False) if val_paths else None

    print("Images:", len(paths), "| Train images:", len(train_paths), "| Val images:", len(val_paths))
    print("Train samples:", len(train_dataset), "| Val samples:", len(val_dataset) if val_dataset else 0)

    # show label distribution
    train_labels = [label for _, label in train_dataset.samples]
    print("Train label distribution:", Counter(train_labels))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

    model = OrientationNet().to(device)

    os.makedirs(model_dir, exist_ok=True)

    best_model_path = os.path.join(model_dir, "orientation_model_best.pth")
    last_model_path = os.path.join(model_dir, "orientation_model_last.pth")

    # optional retraining
    if os.path.exists(best_model_path):
        print("\nLoading existing best model for fine-tuning")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    elif os.path.exists(last_model_path):
        print("\nLoading existing last model for fine-tuning")
        model.load_state_dict(torch.load(last_model_path, map_location=device))

    _unfreeze_last_n_blocks(model, unfreeze_last_n)

    # verify trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("\nTrainable parameters:", len(trainable_params))

    criterion = _make_criterion(label_smoothing=label_smoothing)

    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = -1.0
    no_improve = 0

    for epoch in range(epochs):

        model.train()

        running_loss = 0
        running_correct = 0
        running_total = 0

        loop = tqdm(train_loader, leave=True)

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.numel()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_acc = (running_correct / running_total) if running_total else 0.0

        print(f"\nEpoch {epoch+1}/{epochs} Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_acc = None
        if val_loader is not None:
            val_acc, cm = _evaluate(model, val_loader, device)
            print(f"Val Acc: {val_acc:.4f}")
            if cm is not None:
                print("Val Confusion Matrix (rows=true, cols=pred):\n", cm)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print("Saved best model to:", best_model_path)
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"Early stopping (no val improvement for {patience} epochs).")
                    break

        torch.save(model.state_dict(), last_model_path)

    print("\nLast model saved to:", last_model_path)
    if val_loader is not None and os.path.exists(best_model_path):
        print("Best model saved to:", best_model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--unfreeze_last_n", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)

    args = parser.parse_args()

    train(
        args.data_dir,
        args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        img_size=args.img_size,
        label_smoothing=args.label_smoothing,
        unfreeze_last_n=args.unfreeze_last_n,
        patience=args.patience,
    )
