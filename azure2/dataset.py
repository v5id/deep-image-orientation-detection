import os
import inspect
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Letterbox:

    def __init__(self, size, fill=(255, 255, 255)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image):
        w, h = image.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (self.size, self.size), self.fill)

        scale = min(self.size / w, self.size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = image.resize((nw, nh), resample=Image.BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - nw) // 2
        top = (self.size - nh) // 2
        canvas.paste(resized, (left, top))
        return canvas


class OrientationDataset(Dataset):

    def __init__(self, data_dir=None, paths=None, img_size=224, train=True):

        self.samples = []

        if paths is None:
            if data_dir is None:
                raise ValueError("Either data_dir or paths must be provided")

            paths = []
            for file in os.listdir(data_dir):

                if file.lower().endswith((".jpg", ".jpeg", ".png")):

                    paths.append(os.path.join(data_dir, file))

        for path in paths:
            for label in range(4):
                self.samples.append((path, label))

        def _make_transform(transform_cls, **kwargs):
            sig = inspect.signature(transform_cls.__init__).parameters
            filtered = {k: v for k, v in kwargs.items() if k in sig}
            return transform_cls(**filtered)

        if train:
            self.transform = transforms.Compose([
                Letterbox(img_size),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.25,
                            contrast=0.25,
                            saturation=0.15,
                        )
                    ],
                    p=0.7,
                ),
                _make_transform(
                    transforms.RandomAffine,
                    degrees=2,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05),
                    fill=(255, 255, 255),
                ),
                _make_transform(
                    transforms.RandomPerspective,
                    distortion_scale=0.12,
                    p=0.35,
                    fill=(255, 255, 255),
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                Letterbox(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
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

        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")

        image = self.rotate(image, label)

        image = self.transform(image)

        return image, label
