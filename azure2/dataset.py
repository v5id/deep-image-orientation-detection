import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OrientationDataset(Dataset):

    def __init__(self, data_dir, img_size=224):

        self.samples = []

        for file in os.listdir(data_dir):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                path = os.path.join(data_dir, file)

                for label in range(4):
                    self.samples.append((path, label))


        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.RandomPerspective(
                distortion_scale=0.1,
                p=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def rotate(self, image, label):

        if label == 0:
            return image

        if label == 1:
            return image.rotate(-90, expand=True)

        if label == 2:
            return image.rotate(180, expand=True)

        if label == 3:
            return image.rotate(90, expand=True)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        image = self.rotate(image, label)

        image = self.transform(image)

        return image, label