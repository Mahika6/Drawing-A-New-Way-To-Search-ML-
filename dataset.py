# dataset.py

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ----------------------------------------------------------------
# Cutout class is now defined here so background workers can find it
# ----------------------------------------------------------------
class Cutout:
    """Randomly mask out a square region of the image."""
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):
        h, w = img.size
        # Choose a random center point for the cutout square
        x = random.randint(0, w)
        y = random.randint(0, h)
        
        # Calculate the coordinates of the square
        x1 = max(0, x - self.size // 2)
        y1 = max(0, y - self.size // 2)
        x2 = min(w, x + self.size // 2)
        y2 = min(h, y + self.size // 2)
        
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], fill=0) # Fill with black
        return img

# ----------------------------------------------------------------
# QuickDrawDataset class (as before)
# ----------------------------------------------------------------
class QuickDrawDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Quick, Draw! .npy files.
    """
    def __init__(self, classes, data_dir, samples_per_class=None, transform=None):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        self.transform = transform

        from pathlib import Path
        data_dir = Path(data_dir)

        for c in tqdm(classes, desc="Loading classes"):
            path = data_dir / f"{c}.npy"
            if not path.exists():
                print(f"Warning: File not found and skipped: {path}")
                continue
            
            arr = np.load(path)
            if samples_per_class:
                arr = arr[:samples_per_class]
            
            for a in arr:
                img = a.reshape(28, 28).astype(np.uint8)
                self.samples.append((img, self.class_to_idx[c]))

    # CORRECT
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        pil_img = Image.fromarray(img)
        
        if self.transform:
            x = self.transform(pil_img)
        else:
            x = transforms.ToTensor()(pil_img)

        return x, label