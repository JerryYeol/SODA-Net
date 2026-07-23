"""
PyTorch version of Loge Loss
Strictly converted from TensorFlow implementation
"""

import torch
import torch.nn as nn
import numpy as np


class LogeLoss(nn.Module):
    """
    Loge Loss: Logarithmic variant of cross-entropy loss

    Reference from TensorFlow implementation:
    loss = log(cross_entropy_loss + eps) - log(eps)

    where eps = 1 - log(2.0)
    """
    def __init__(self, bias=1e-10, eps=None):
        super(LogeLoss, self).__init__()

        self.bias = bias

        # Default eps = 1 - log(2.0)
        if eps is None:
            self.eps = torch.tensor(1.0 - np.log(2.0), dtype=torch.float32)
        else:
            self.eps = torch.tensor(eps, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        """
        Compute Loge loss

        Args:
            y_pred: predicted probabilities (batch, num_classes)
                    Should be output of softmax or log_softmax
            y_true: true labels (batch,) as class indices
                    or (batch, num_classes) as one-hot

        Returns:
            loss: scalar loss value
        """
        # Convert y_true to one-hot if needed
        if y_true.dim() == 1:
            # y_true is class indices: (batch,)
            num_classes = y_pred.shape[1]
            y_true_onehot = torch.zeros_like(y_pred)
            y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1.0)
        else:
            # y_true is already one-hot: (batch, num_classes)
            y_true_onehot = y_true

        # Ensure y_pred is on the same device as y_true
        self.eps = self.eps.to(y_pred.device)

        # Apply softmax if y_pred is not in probability space
        # Assume y_pred is logits, apply softmax
        if y_pred.requires_grad:
            y_pred_prob = torch.softmax(y_pred, dim=-1)
        else:
            y_pred_prob = y_pred

        # Compute cross-entropy loss manually
        # cross_entropy = - sum(y_true * log(y_pred + bias))
        cross_entropy_loss = -torch.sum(y_true_onehot * torch.log(y_pred_prob + self.bias), dim=-1)

        # Apply logarithmic transformation
        # loss = log(cross_entropy + eps) - log(eps)
        loss = torch.log(cross_entropy_loss + self.eps) - torch.log(self.eps)

        # Return mean loss (TensorFlow loss class automatically does reduce_mean)
        return loss.mean()
