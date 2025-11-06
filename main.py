import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import sys
import json
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# --- Robust imports (root or utils/) ---
try:
    from utils.dataloder import CTScanDataset
except ModuleNotFoundError:
    from utils.dataloder import CTScanDataset  # noqa

try:
    from utils.augmentation import get_augmentation
except ModuleNotFoundError:
    from utils.augmentation import get_augmentation  # noqa

try:
    from utils.evaluation import ModelTrainer
except ModuleNotFoundError:
    from utils.evaluation import ModelTrainer  # noqa

try:
    from models.modelengine import get_model
except ModuleNotFoundError:
    from models.modelengine import get_model  # noqa


def parse_args():
    p = argparse.ArgumentParser(description="Retinal/CT Disease Classification CLI (RGB/Gray)")
    p.add_argument('--data_path', type=str, required=True, help='Root folder with class subfolders')
    p.add_argument('--model', type=str, default='densenet121',
                   choices=['customcnn', 'mobilenetv3', 'densenet121'])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'])
    p.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'step'])
    p.add_argument('--aug_type', type=str, default='mild', choices=['none', 'mild', 'strong', 'advanced'])
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--color_mode', type=str, default='rgb', choices=['rgb', 'gray'],
                   help='Input color mode; use gray for single-channel inputs')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split_indices(labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    idx = np.arange(len(labels))
    y = np.array(labels)
    from sklearn.model_selection import train_test_split
    train_idx, temp_idx = train_test_split(idx, train_size=train_ratio, stratify=y, random_state=seed)
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_size, stratify=y[temp_idx], random_state=seed)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def subset_dataset(dataset: CTScanDataset, indices, transform):
    ds = CTScanDataset(dataset.data_dir, transform=transform, subset='train', color_mode=dataset.color_mode)
    ds.class_to_idx = dataset.class_to_idx
    ds.idx_to_class = dataset.idx_to_class
    ds.samples = [dataset.samples[i] for i in indices]
    return ds


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---------- Result/<model_name>_<aug_type>/ ----------
    run_name = f"{args.model}_{args.aug_type}".lower()
    # sanitize to be file-system friendly
    run_name = re.sub(r'[^a-z0-9_.-]+', '-', run_name)

    result_root = os.path.join("Result", run_name)
    os.makedirs(result_root, exist_ok=True)
    args.save_dir = os.path.join(result_root, "checkpoints")
    args.log_dir = os.path.join(result_root, "logs")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Save config early
    with open(os.path.join(result_root, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    # -----------------------------------------------------

    if not os.path.isdir(args.data_path):
        print(f"❌ data_path not found: {args.data_path}")
        sys.exit(1)

    in_channels = 3 if args.color_mode.lower() == 'rgb' else 1

    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model}")
    print(f"Augmentation: {args.aug_type}")
    print(f"Image size: {args.img_size}")
    print(f"Color mode: {args.color_mode} ({in_channels} ch)")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"LR: {args.lr} | Weight Decay: {args.weight_decay}")
    print(f"Optimizer: {args.optimizer} | Scheduler: {args.scheduler}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Artifacts root: {result_root}")
    print("=" * 70)

    # Transforms (channel-aware normalization)
    train_tf = get_augmentation(args.aug_type, args.img_size, in_channels=in_channels)
    eval_tf = get_augmentation("none", args.img_size, in_channels=in_channels)

    # Read full dataset
    full_ds = CTScanDataset(data_dir=args.data_path, transform=None, subset='full', color_mode=args.color_mode)
    labels = [lbl for _, lbl in full_ds.samples]
    class_names = [name for name, _ in sorted(full_ds.class_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(class_names)
    print(f"Detected classes ({num_classes}): {class_names}")

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split_indices(labels, 0.8, 0.1, args.seed)

    # Final datasets
    train_ds = subset_dataset(full_ds, train_idx, train_tf)
    val_ds = subset_dataset(full_ds, val_idx, eval_tf)
    test_ds = subset_dataset(full_ds, test_idx, eval_tf)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False)

    # Model (channel-aware)
    model = get_model(args.model, num_classes=num_classes, in_channels=in_channels, pretrained=args.pretrained)
    model.to(args.device)

    # Class weights (inverse frequency)
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = (len(labels) / (num_classes * np.clip(class_counts, 1, None))).astype(np.float32)
    class_weights = torch.tensor(weights, device=args.device)

    # Trainer
    trainer = ModelTrainer(
        model=model,
        device=torch.device(args.device),
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        class_names=class_names
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        use_scheduler=True,
        patience=10
    )

    # Test
    print("Evaluating on test set…")
    results = trainer.evaluate(test_loader)
    print("Final Test Metrics:")
    for k, v in results.items():
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Persist final metrics
    with open(os.path.join(result_root, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\n✅ Artifacts saved under: {result_root}")
    print(f"   - Checkpoints : {args.save_dir}")
    print(f"   - Logs        : {args.log_dir}")
    print(f"   - Config      : {os.path.join(result_root, 'config.json')}")
    print(f"   - Test metrics: {os.path.join(result_root, 'test_metrics.json')}")


if __name__ == "__main__":
    main()