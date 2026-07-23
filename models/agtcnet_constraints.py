"""
PyTorch version of AGTCNet constraints
Strictly converted from TensorFlow implementation
"""

import torch
import torch.nn as nn


class MinMaxValue:
    """
    Constraint to clip weights between min_value and max_value
    Equivalent to TensorFlow's tf.clip_by_value
    """
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        """
        Clips weight tensor to [min_value, max_value]

        Args:
            w: weight tensor

        Returns:
            clipped weight tensor
        """
        return torch.clamp(w, min=self.min_value, max=self.max_value)
