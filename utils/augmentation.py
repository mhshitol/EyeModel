# aumentation.py
# Different augmentation levels for retinal images; supports 1-ch (gray) or 3-ch (RGB)

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _norm_stats(in_channels: int):
    if in_channels == 3:
        return IMAGENET_MEAN, IMAGENET_STD
    else:
        # reasonable grayscale stats
        return [0.5], [0.5]

def get_augmentation(aug_type: str = "none", image_size: int = 224, in_channels: int = 3):
    """
    Args:
        aug_type: ['none', 'mild', 'strong', 'advanced']
        image_size: output square size
        in_channels: 1 or 3
    """
    mean, std = _norm_stats(in_channels)
    aug_type = aug_type.lower()

    base = [A.Resize(image_size, image_size)]

    if aug_type == "none":
        tfm = base + [A.Normalize(mean=mean, std=std), ToTensorV2()]
    elif aug_type == "mild":
        tfm = base + [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    elif aug_type == "strong":
        tfm = base + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05, p=0.4) if in_channels==3 else A.NoOp(),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    elif aug_type == "advanced":
        tfm = base + [
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
            A.HueSaturationValue(p=0.4) if in_channels==3 else A.NoOp(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

    # filter out NoOp for grayscale mode
    tfm = [t for t in tfm if not isinstance(t, A.NoOp)]
    return A.Compose(tfm)
