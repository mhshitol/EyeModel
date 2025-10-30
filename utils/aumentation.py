# Apply differences type of data augmentation techniques
# aumentation.py
# Different augmentation levels for grayscale fundus images (1-channel)

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation(aug_type: str = "none", image_size: int = 224):
    """
    Args:
        aug_type: ['none', 'mild', 'strong', 'advanced']
        image_size: output square size
    Returns:
        Albumentations Compose object producing 1-channel tensors
    """
    aug_type = aug_type.lower()

    if aug_type == "none":
        tfm = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ]

    elif aug_type == "mild":
        tfm = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ]

    elif aug_type == "strong":
        tfm = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ]

    elif aug_type == "advanced":
        tfm = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.4),
            A.OneOf([
                A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.6),
                A.Sharpen(alpha=(0.1, 0.2), lightness=(0.9, 1.0), p=0.4),
            ], p=0.6),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.2),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ]

    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

    return A.Compose(tfm)
