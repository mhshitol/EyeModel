# evaluation.py (REPLACE your file with this updated version)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
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
    """Generic N-class metrics with optional class names."""
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        if class_names is not None:
            self.class_names = list(class_names)
        else:
            self.class_names = [f"Class_{i}" for i in range(num_classes)]

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_probs: Optional[np.ndarray] = None) -> Dict:
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
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }

        # Per-class metrics + sensitivity/specificity one-vs-all for ALL classes
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

            key = name.lower().replace(' ', '_')
            metrics[f'{key}_precision'] = p
            metrics[f'{key}_recall'] = r
            metrics[f'{key}_f1'] = f
            metrics[f'{key}_sensitivity'] = sensitivity
            metrics[f'{key}_specificity'] = specificity

        # AUC-ROC (macro OVR if probs provided)
        if y_probs is not None:
            try:
                if self.num_classes == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    metrics['auc_roc_macro'] = roc_auc_score(
                        y_true, y_probs, multi_class='ovr', average='macro'
                    )
            except ValueError:
                logger.warning("Could not calculate AUC-ROC score")

        return metrics

class ModelTrainer:
    def __init__(self, model: nn.Module, device: torch.device,
                 save_dir: str = 'checkpoints', log_dir: str = 'logs',
                 class_names: Optional[List[str]] = None):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize with placeholder; will set proper num_classes in train/evaluate
        self.metrics_calculator = None
        self.class_names = class_names

        self.writer = SummaryWriter(log_dir=log_dir)
        self.history = defaultdict(list)

    def _ensure_metrics(self, num_classes: int):
        if self.metrics_calculator is None:
            self.metrics_calculator = ModelMetrics(num_classes=num_classes, class_names=self.class_names)

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, epoch: int) -> Dict:
        self.model.train()
        running_loss = 0.0
        all_predictions, all_labels, all_probabilities = [], [], []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.detach().cpu().numpy())
            all_labels.extend(target.detach().cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())

            progress_bar.set_postfix({'Loss': loss.item()})
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        avg_loss = running_loss / len(train_loader)
        self._ensure_metrics(num_classes=outputs.shape[1])
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        return metrics

    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module, epoch: int) -> Dict:
        self.model.eval()
        running_loss = 0.0
        all_predictions, all_labels, all_probabilities = [], [], []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Validation')
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                loss = criterion(outputs, target)
                running_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.detach().cpu().numpy())
                all_labels.extend(target.detach().cpu().numpy())
                all_probabilities.extend(probabilities.detach().cpu().numpy())

                progress_bar.set_postfix({'Loss': loss.item()})

        avg_loss = running_loss / len(val_loader)
        self._ensure_metrics(num_classes=outputs.shape[1])
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, class_weights: Optional[torch.Tensor] = None,
              use_scheduler: bool = True, patience: int = 10,
              criterion: Optional[nn.Module] = None,
              optimizer_name: str = 'adamw',
              scheduler_name: str = 'plateau') -> Dict:

        optimizer = get_optimizer(self.model, optimizer_name, learning_rate, weight_decay)
        scheduler = get_scheduler(optimizer, scheduler_name) if use_scheduler else None

        if criterion is None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)

        best_val_loss = float('inf')
        best_model_state = None
        early_stopping = EarlyStopping(patience=patience, mode='min')

        logger.info(f"Starting training loop with optimizer={optimizer_name}, scheduler={scheduler_name if use_scheduler else 'None'}")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch)
            # Validate
            val_metrics = self.validate_epoch(val_loader, criterion, epoch)
            val_loss = val_metrics['loss']

            # Scheduler step
            if scheduler is not None:
                if scheduler_name.lower() == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                self.save_checkpoint(epoch, {'loss': train_metrics['loss'], 'accuracy': train_metrics['accuracy']}, val_metrics, is_best=True)
            elif early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Best model weights loaded.")

        logger.info("Training loop completed.")
        self.writer.close()

        self.plot_training_history(metrics=['loss'], save_path=os.path.join(self.save_dir, 'loss_curve.png'))
        self.plot_training_history(metrics=['accuracy'], save_path=os.path.join(self.save_dir, 'accuracy_curve.png'))

        return self.history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        self.model.eval()
        all_predictions, all_labels, all_probabilities = [], [], []

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Testing')
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        self._ensure_metrics(num_classes=outputs.shape[1])
        test_metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )

        self.plot_confusion_matrix(all_labels, all_predictions)
        self._generate_classification_report(test_metrics)

        return test_metrics

    def plot_confusion_matrix(self, y_true: List, y_pred: List, save_path: str = None):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.metrics_calculator.num_classes)))
        plt.figure(figsize=(1.0 + 0.6*len(self.metrics_calculator.class_names),
                            1.0 + 0.6*len(self.metrics_calculator.class_names)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.metrics_calculator.class_names,
                    yticklabels=self.metrics_calculator.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_training_history(self, metrics: List[str] = ['loss', 'accuracy'], save_path: str = None):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        for i, metric in enumerate(metrics):
            tkey = f'train_{metric}'
            vkey = f'val_{metric}'
            if tkey in self.history and vkey in self.history:
                axes[i].plot(self.history[tkey], label=f'Train {metric}')
                axes[i].plot(self.history[vkey], label=f'Val {metric}')
                axes[i].set_title(f'{metric.capitalize()} History')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        path = os.path.join(self.save_dir, 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        for k, v in train_metrics.items():
            self.writer.add_scalar(f'Train/{k}', v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)
        logger.info(f"Epoch {epoch+1}: Train Loss {train_metrics['loss']:.4f} Acc {train_metrics['accuracy']:.4f} | "
                    f"Val Loss {val_metrics['loss']:.4f} Acc {val_metrics['accuracy']:.4f} F1 {val_metrics['f1_macro']:.4f}")

def get_optimizer(model: nn.Module, optimizer_name: str = 'adamw',
                  learning_rate: float = 0.001, weight_decay: float = 1e-4) -> optim.Optimizer:
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str = 'plateau'):
    if scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    elif scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
