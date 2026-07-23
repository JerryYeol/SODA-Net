"""
PyTorch version of AGTCNet positional encoding
Strictly converted from TensorFlow implementation
"""

import torch
import torch.nn as nn
import math
from .agtcnet_constraints import MinMaxValue


class PositionalEncoding(nn.Module):
    """
    Positional Encoding layer with learnable scale
    Modified version of standard positional encoding
    """
    def __init__(self,
                 trainable_scale=True,
                 scale_initializer='zeros',
                 scale_constraint=None):
        super(PositionalEncoding, self).__init__()

        self.trainable_scale = trainable_scale
        self.scale_constraint = scale_constraint

        # Learnable scale parameter
        if trainable_scale:
            self.scale = nn.Parameter(torch.zeros(1))
            if scale_initializer == 'zeros':
                nn.init.zeros_(self.scale)
            elif scale_initializer == 'ones':
                nn.init.ones_(self.scale)
        else:
            self.register_buffer('scale', torch.zeros(1))

    def _positional_encoding(self, max_pos, embedding_dim, dtype=torch.float32):
        """
        Generate positional encoding

        Args:
            max_pos: maximum position (time steps)
            embedding_dim: embedding dimension (features)
            dtype: data type

        Returns:
            positional encoding tensor of shape (max_pos, embedding_dim)
        """
        # positions: (T, 1)
        positions = torch.arange(max_pos, dtype=dtype).unsqueeze(1)

        # embedding_idx: (1, F)
        embedding_idx = torch.arange(embedding_dim, dtype=dtype).unsqueeze(0)

        # power = 2 * (embedding_idx // 2) / embedding_dim
        power = 2 * (embedding_idx // 2) / embedding_dim

        # angle_rates = 1 / (10000 ** power)
        angle_rates = 1.0 / (10000.0 ** power)

        # angle_rads = positions * angle_rates
        # (T, 1) * (1, F) = (T, F)
        angle_rads = positions * angle_rates

        # Apply sin to even indices, cos to odd indices
        # TensorFlow uses broadcasting with modulo
        encoding = torch.zeros(max_pos, embedding_dim, dtype=dtype)

        # Even indices: sin
        encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # Odd indices: cos
        encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return encoding

    def forward(self, x):
        """
        Add positional encoding to input

        Args:
            x: input tensor of shape (batch, time, features)

        Returns:
            x + scaled positional encoding
        """
        # Apply constraint if specified
        if self.scale_constraint is not None:
            with torch.no_grad():
                self.scale.data = self.scale_constraint(self.scale.data)

        batch_size, max_pos, embedding_dim = x.shape

        # Generate positional encoding
        # (T, F)
        pos_encoding = self._positional_encoding(max_pos, embedding_dim, dtype=x.dtype)

        # Move to same device as input
        pos_encoding = pos_encoding.to(x.device)

        # Add positional encoding scaled by learnable parameter
        # Broadcasting: (B, T, F) + (T, F) = (B, T, F)
        output = x + self.scale * pos_encoding

        return output


class Scale(nn.Module):
    """
    Learnable scalar multiplication layer
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
