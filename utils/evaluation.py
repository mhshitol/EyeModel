# utils/evaluation.py
# Comprehensive training/evaluation utilities for N-class retinal classification

import os
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------- Utilities ---------------------------- #

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


class ModelMetrics:
    """Generic N-class metrics (accuracy, macro/weighted P/R/F1, per-class metrics, AUC)."""
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        if class_names is not None:
            self.class_names = list(class_names)
        else:
            self.class_names = [f"Class_{i}" for i in range(num_classes)]

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None
    ) -> Dict:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
        }

        # Per-class metrics + sensitivity/specificity (one-vs-all)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        for i, name in enumerate(self.class_names):
            p = precision[i] if i < len(precision) else 0.0
            r = recall[i] if i < len(recall) else 0.0
            f = f1[i] if i < len(f1) else 0.0

            tp = cm[i, i] if i < cm.shape[0] else 0
            fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
            fp = cm[:, i].sum() - tp if i < cm.shape[0] else 0
            tn = cm.sum() - (tp + fn + fp)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            key = name.lower().replace(" ", "_")
            metrics[f"{key}_precision"] = float(p)
            metrics[f"{key}_recall"] = float(r)
            metrics[f"{key}_f1"] = float(f)
            metrics[f"{key}_sensitivity"] = float(sensitivity)
            metrics[f"{key}_specificity"] = float(specificity)

        # AUC-ROC (macro OVR) if probabilities provided
        if y_probs is not None:
            try:
                if self.num_classes == 2:
                    metrics["auc_roc"] = float(roc_auc_score(y_true, y_probs[:, 1]))
                else:
                    metrics["auc_roc_macro"] = float(
                        roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
                    )
            except ValueError:
                logger.warning("Could not calculate AUC-ROC score")

        return metrics


# ---------------------------- Trainer ---------------------------- #

class ModelTrainer:
    """Main training/evaluation engine."""
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.metrics_calculator: Optional[ModelMetrics] = None
        self.class_names = class_names

        self.writer = SummaryWriter(log_dir=log_dir)
        self.history = defaultdict(list)

    # ---- helpers ----
    def _ensure_metrics(self, num_classes: int):
        if self.metrics_calculator is None:
            self.metrics_calculator = ModelMetrics(num_classes=num_classes, class_names=self.class_names)

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        for k, v in train_metrics.items():
            self.writer.add_scalar(f"Train/{k}", v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"Val/{k}", v, epoch)
        logger.info(
            f"Epoch {epoch+1}: Train Loss {train_metrics.get('loss', 0):.4f} "
            f"Acc {train_metrics.get('accuracy', 0):.4f} | "
            f"Val Loss {val_metrics.get('loss', 0):.4f} "
            f"Acc {val_metrics.get('accuracy', 0):.4f} "
            f"F1 {val_metrics.get('f1_macro', 0):.4f}"
        )

    # ---- epoch loops ----
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> Dict:
        self.model.train()
        running_loss = 0.0
        all_predictions, all_labels, all_probabilities = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)

            all_predictions.extend(pred.detach().cpu().numpy())
            all_labels.extend(target.detach().cpu().numpy())
            all_probabilities.extend(probs.detach().cpu().numpy())

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

        avg_loss = running_loss / max(1, len(train_loader))
        self._ensure_metrics(num_classes=outputs.shape[1])
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )
        metrics["loss"] = float(avg_loss)
        return metrics

    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module, epoch: int) -> Dict:
        self.model.eval()
        running_loss = 0.0
        all_predictions, all_labels, all_probabilities = [], [], []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                loss = criterion(outputs, target)
                running_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)

                all_predictions.extend(pred.detach().cpu().numpy())
                all_labels.extend(target.detach().cpu().numpy())
                all_probabilities.extend(probs.detach().cpu().numpy())

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / max(1, len(val_loader))
        self._ensure_metrics(num_classes=outputs.shape[1])
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )
        metrics["loss"] = float(avg_loss)
        return metrics

    # ---- full training ----
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        use_scheduler: bool = True,
        patience: int = 10,
        criterion: Optional[nn.Module] = None,
        optimizer_name: str = "adamw",
        scheduler_name: str = "plateau",
    ) -> Dict:
        optimizer = get_optimizer(self.model, optimizer_name, learning_rate, weight_decay)
        scheduler = get_scheduler(optimizer, scheduler_name) if use_scheduler else None

        if criterion is None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)

        best_val_loss = float("inf")
        best_state = None
        early = EarlyStopping(patience=patience, mode="min")

        logger.info(
            f"Starting training with optimizer={optimizer_name}, scheduler={scheduler_name if use_scheduler else 'None'}"
        )

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch)
            val_metrics = self.validate_epoch(val_loader, criterion, epoch)
            val_loss = val_metrics["loss"]

            # Scheduler step
            if scheduler is not None:
                if scheduler_name.lower() == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Logging
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics.get("accuracy", 0.0))
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
            self.history["val_f1_macro"].append(val_metrics.get("f1_macro", 0.0))

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict()
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
            elif early(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Loaded best model weights.")

        self.writer.close()
        self.plot_training_history(metrics=["loss"], save_path=os.path.join(self.save_dir, "loss_curve.png"))
        self.plot_training_history(metrics=["accuracy"], save_path=os.path.join(self.save_dir, "accuracy_curve.png"))
        return self.history

    # ---- evaluation ----
    def evaluate(self, test_loader: DataLoader) -> Dict:
        self.model.eval()
        all_predictions, all_labels, all_probabilities = [], [], []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)

                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

        # Ensure metrics calculator is ready
        self._ensure_metrics(num_classes=outputs.shape[1])
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )

        # Plots + report
        self.plot_confusion_matrix(all_labels, all_predictions)
        self._generate_classification_report(metrics)
        return metrics

    # ---- visualization & reports ----
    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict, is_best: bool = False):
        """
        Save a lightweight checkpoint with model weights and a few metrics.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        ckpt = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "train_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in train_metrics.items()},
            "val_metrics":   {k: float(v) if isinstance(v, (int, float)) else v for k, v in val_metrics.items()},
        }
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth"
        path = os.path.join(self.save_dir, filename)
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from a checkpoint file created by save_checkpoint.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Checkpoint loaded: {checkpoint_path} (epoch {ckpt.get('epoch', 'n/a')})")
        return ckpt

    def plot_confusion_matrix(self, y_true: List, y_pred: List, save_path: Optional[str] = None):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.metrics_calculator.num_classes)))
        n = len(self.metrics_calculator.class_names)
        plt.figure(figsize=(1.0 + 0.6 * n, 1.0 + 0.6 * n))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.metrics_calculator.class_names,
            yticklabels=self.metrics_calculator.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        if save_path is None:
            save_path = os.path.join(self.save_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_training_history(self, metrics: List[str] = ["loss", "accuracy"], save_path: Optional[str] = None):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        for i, metric in enumerate(metrics):
            tkey = f"train_{metric}"
            vkey = f"val_{metric}"
            if tkey in self.history and vkey in self.history:
                axes[i].plot(self.history[tkey], label=f"Train {metric}")
                axes[i].plot(self.history[vkey], label=f"Val {metric}")
                axes[i].set_title(f"{metric.capitalize()} History")
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.save_dir, "training_history.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_classification_report(self, metrics: Dict):
        """
        Save a compact, human-readable classification report to JSON and log a short summary.
        Expects overall keys (accuracy, f1_macro, f1_weighted, etc.) and per-class keys
        like '<class>_precision', '<class>_recall', '<class>_f1', '<class>_sensitivity', '<class>_specificity'.
        """
        # Resolve class names
        if self.metrics_calculator is not None and getattr(self.metrics_calculator, "class_names", None):
            class_names = self.metrics_calculator.class_names
        else:
            # Try to infer
            class_names = []
            for k in metrics.keys():
                if k.endswith("_precision"):
                    base = k[: -len("_precision")]
                    class_names.append(base.replace("_", " ").title())
            class_names = sorted(set(class_names)) if class_names else [f"Class_{i}" for i in range(1)]

        report = {
            "overall": {
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "macro_f1": float(metrics.get("f1_macro", 0.0)),
                "weighted_f1": float(metrics.get("f1_weighted", 0.0)),
                "precision_macro": float(metrics.get("precision_macro", 0.0)),
                "recall_macro": float(metrics.get("recall_macro", 0.0)),
            },
            "per_class": {},
        }

        for name in class_names:
            key = name.lower().replace(" ", "_")
            report["per_class"][name] = {
                "precision": float(metrics.get(f"{key}_precision", 0.0)),
                "recall": float(metrics.get(f"{key}_recall", 0.0)),
                "f1": float(metrics.get(f"{key}_f1", 0.0)),
                "sensitivity": float(metrics.get(f"{key}_sensitivity", 0.0)),
                "specificity": float(metrics.get(f"{key}_specificity", 0.0)),
            }

        os.makedirs(self.save_dir, exist_ok=True)
        out_path = os.path.join(self.save_dir, "classification_report.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Classification report saved: {out_path}")
        except Exception as e:
            logger.warning(f"Could not write classification report: {e}")

        logger.info("=== CLASSIFICATION REPORT (summary) ===")
        logger.info(
            f"Accuracy: {report['overall']['accuracy']:.4f} | "
            f"Macro F1: {report['overall']['macro_f1']:.4f} | "
            f"Weighted F1: {report['overall']['weighted_f1']:.4f}"
        )
        for name, vals in report["per_class"].items():
            logger.info(f"{name}: F1={vals['f1']:.4f}  P={vals['precision']:.4f}  R={vals['recall']:.4f}")


# ---------------------------- Optimizers / Schedulers ---------------------------- #

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    name = optimizer_name.lower()
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str = "plateau"):
    name = scheduler_name.lower()
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
