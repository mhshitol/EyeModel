# modelengine.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.customcnn import CustomCNN
from models.mobilenetv3 import MobileNetV3
from models.densenet121 import DenseNet121Medical

def get_model(model_name: str, num_classes: int, in_channels: int = 3,
              pretrained: bool = True, dropout_rate: float = 0.5):
    if model_name == 'customcnn':
        return CustomCNN(num_classes=num_classes, in_ch=in_channels)
    elif model_name == "mobilenetv3":
        return MobileNetV3(num_classes=num_classes, in_ch=in_channels,
                           pretrained=pretrained, dropout_rate=dropout_rate)
    elif model_name == "densenet121":
        return DenseNet121Medical(num_classes=num_classes, in_ch=in_channels,
                                  pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Model {model_name} is not supported.")


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else (loss.sum() if self.reduction == 'sum' else loss)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes; self.smoothing = smoothing; self.confidence = 1.0 - smoothing
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        tgt_1h = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        tgt_s = tgt_1h * self.confidence + (1 - tgt_1h) * self.smoothing / (self.num_classes - 1)
        return (-tgt_s * log_probs).sum(dim=1).mean()
