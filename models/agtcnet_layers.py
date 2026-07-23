"""
PyTorch version of AGTCNet layers
Strictly converted from TensorFlow implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .agtcnet_constraints import MinMaxValue


class Reshape(nn.Module):
    """
    Reshape layer
    """
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(self.target_shape)


class UnscaledDropout(nn.Module):
    """
    Dropout without scaling (keeps original values without 1/(1-p) scaling)
    TensorFlow equivalent: tf.random.uniform + mask multiplication
    """
    def __init__(self, drop_rate):
        super(UnscaledDropout, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            # Generate random mask: 1 if random >= drop_rate, else 0
            mask = torch.rand_like(x) >= self.drop_rate
            # Multiply without scaling
            output = x * mask.float()
            return output
        else:
            return x


class WeightedAdd(nn.Module):
    """
    Weighted addition of multiple tensors
    Equivalent to TensorFlow's Add([arr * weight[i] for i, arr in enumerate(input)])
    """
    def __init__(self,
                 input_size,
                 weight_initializer='zeros',
                 weight_constraint=None):
        super(WeightedAdd, self).__init__()

        self.input_size = input_size
        self.weight_constraint = weight_constraint

        # Initialize weight
        self.weight = nn.Parameter(torch.zeros(input_size))

        if weight_initializer == 'zeros':
            nn.init.zeros_(self.weight)
        elif weight_initializer == 'ones':
            nn.init.ones_(self.weight)

    def forward(self, inputs):
        """
        Args:
            inputs: list of tensors with same shape

        Returns:
            weighted sum of inputs
        """
        # Apply constraint if specified
        if self.weight_constraint is not None:
            with torch.no_grad():
                self.weight.data = self.weight_constraint(self.weight.data)

        # Weighted addition
        output = sum([arr * self.weight[i] for i, arr in enumerate(inputs)])
        return output


class Scale(nn.Module):
    """
    Learnable scalar multiplication layer
    Equivalent to TensorFlow's scale * inputs
    """
    def __init__(self,
                 initializer='zeros',
                 constraint=None):
        super(Scale, self).__init__()

        self.constraint = constraint

        # Initialize scale as a learnable parameter
        self.scale = nn.Parameter(torch.zeros(1))

        if initializer == 'zeros':
            nn.init.zeros_(self.scale)
        elif initializer == 'ones':
            nn.init.ones_(self.scale)

    def forward(self, x):
        """
        Args:
            x: input tensor

        Returns:
            scaled tensor
        """
        # Apply constraint if specified
        if self.constraint is not None:
            with torch.no_grad():
                self.scale.data = self.constraint(self.scale.data)

        return self.scale * x
