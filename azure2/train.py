import argparse
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OrientationDataset
from model import OrientationNet


def train(data_dir, model_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    dataset = OrientationDataset(data_dir)

    print("Dataset size:", len(dataset))

    # show label distribution
    labels = [label for _, label in dataset.samples]
    print("Label distribution:", Counter(labels))

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    model = OrientationNet().to(device)

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "orientation_model.pth")

    # optional retraining
    if os.path.exists(model_path):
        print("\nLoading existing model for retraining")
        model.load_state_dict(torch.load(model_path, map_location=device))

    # verify trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("\nTrainable parameters:", len(trainable_params))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        trainable_params,
        lr=0.001
    )

    epochs = 50

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        loop = tqdm(dataloader, leave=True)

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)

        print(f"\nEpoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

        # quick prediction check
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)[:10]
            print("Sample predictions:", preds.cpu().numpy())

    torch.save(model.state_dict(), model_path)

    print("\nModel saved to:", model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)

    args = parser.parse_args()

    train(args.data_dir, args.model_dir)