# dataloder.py
# Load dataset and prepare dataloaders (now supports rgb/gray)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTScanDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, subset: str = 'train', color_mode: str = 'rgb'):
        """
        Args:
            data_dir: Root directory containing class folders
            transform: Albumentations transform pipeline
            subset: 'train', 'val', 'test', or 'full'
            color_mode: 'rgb' or 'gray'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset
        self.color_mode = color_mode.lower()

        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        self.samples = self._load_samples()
        self.class_weights = self._calculate_class_weights()

        logger.info(f"Loaded {len(self.samples)} samples for {subset} set")
        self._print_class_distribution()

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path) or class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):
                    samples.append((os.path.join(class_path, img_name), class_idx))
        return samples

    def _calculate_class_weights(self) -> torch.Tensor:
        class_counts = [0] * len(self.class_to_idx)
        for _, label in self.samples:
            class_counts[label] += 1
        total_samples = len(self.samples)
        weights = []
        for count in class_counts:
            if count == 0:
                weights.append(0.0)
                logger.warning("⚠️ A class has 0 samples — check dataset.")
            else:
                weights.append(total_samples / (len(self.class_to_idx) * count))
        return torch.FloatTensor(weights)

    def _print_class_distribution(self):
        counts = [0] * len(self.class_to_idx)
        for _, lab in self.samples:
            counts[lab] += 1
        logger.info("Class distribution:")
        for idx, c in enumerate(counts):
            logger.info(f"  {self.idx_to_class[idx]}: {c} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = self._load_and_preprocess_image(img_path)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image, label

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        try:
            if self.color_mode == 'rgb':
                # BGR -> RGB
                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Could not load image: {img_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # CLAHE on L channel in LAB (common for fundus contrast)
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
                return rgb.astype(np.uint8)

            else:  # grayscale path retained for compatibility
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image = clahe.apply(image)
                return image.astype(np.uint8)

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            if self.color_mode == 'rgb':
                return np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                return np.zeros((224, 224), dtype=np.uint8)
