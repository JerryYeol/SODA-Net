"""
PyTorch version of AGTCNet
Strictly converted from TensorFlow implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .agtcnet_gcat import GCAT
from .agtcnet_positional_encoding import PositionalEncoding
from .agtcnet_layers import UnscaledDropout, WeightedAdd, Scale
from .agtcnet_constraints import MinMaxValue


class AGTCNet(nn.Module):
    """
    Adaptive Graph Temporal Convolutional Network for EEG Classification

    Args:
        n_classes: number of classes
        Chans: number of EEG channels
        Samples: number of time samples
        F1: number of temporal filters
        D: depth multiplier for depthwise convolution
        kernLength: length of temporal kernel
        dropout: dropout rate
    """
    def __init__(self, n_classes=2, Chans=22, Samples=1001,
                 F1=16, D=2, kernLength=32, dropout=0.5, feature_dim=None,
                 disable_gcat=False, disable_tce=False):
        super(AGTCNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.F1 = F1
        self.D = D
        self.kernLength = kernLength
        self.dropout_rate = dropout
        self.feature_dim = feature_dim  # Optional output feature dimension
        self.disable_gcat = disable_gcat  # Ablation: disable graph attention
        self.disable_tce = disable_tce    # Ablation: disable temporal context enhancement

        F2 = F1 * D  # Number of features after depthwise conv

        # ============================================================
        # Channel-wise Temporal Convolution (CTC) Module
        # ============================================================
        # Input: (batch, 1, Chans, Samples)
        # TensorFlow: Conv2D(F1, (1, kernLength), padding='same', use_bias=False)
        self.ctc_conv = nn.Conv2d(1, F1, (1, kernLength),
                                  padding=(0, kernLength // 2), bias=False)
        self.ctc_bn = nn.BatchNorm2d(F1)

        # ============================================================
        # Local Spatial Feature Extraction (LSFE) Module
        # ============================================================
        # Depthwise convolution: one filter per input channel
        # TensorFlow: DepthwiseConv2D((Chans, 1), depth_multiplier=D, use_bias=False)
        self.lsfe_depthwise = nn.Conv2d(F1, F2, (Chans, 1),
                                        groups=F1, bias=False)
        self.lsfe_bn = nn.BatchNorm2d(F2)
        self.lsfe_activation = nn.ELU()
        self.lsfe_pool = nn.AvgPool2d((1, 4))
        self.lsfe_dropout = nn.Dropout(dropout)

        # ============================================================
        # Local Temporal Feature Extraction (LTFE) Module
        # ============================================================
        # Separable convolution
        # TensorFlow: SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)
        # Separable = Depthwise + Pointwise
        self.ltfe_depthwise = nn.Conv2d(F2, F2, (1, 16),
                                        padding=(0, 8), groups=F2, bias=False)
        self.ltfe_pointwise = nn.Conv2d(F2, F2, 1, bias=False)
        self.ltfe_bn = nn.BatchNorm2d(F2)
        self.ltfe_activation = nn.ELU()
        self.ltfe_pool = nn.AvgPool2d((1, 8))
        self.ltfe_dropout = nn.Dropout(dropout)

        # Calculate size after convolutions and pooling
        # After lsfe_pool: Samples // 4
        # After ltfe_pool: Samples // 4 // 8 = Samples // 32
        self.temporal_len = Samples // 32
        self.feature_dim = F2

        # ============================================================
        # Graph Construction Adaptive Transform (GCAT) Module
        # ============================================================
        # Hybrid GCAT: Fixed graph structure + learnable scaling
        # Treat F2 feature channels as graph nodes

        self.gcat_channels = F2
        self.gcat_heads = 2  # 2 attention heads for efficiency

        # Build graph structure: Self-loops + Nearest neighbors
        # This creates a local connectivity pattern among feature channels
        adj_init = torch.eye(F2)  # Self-connections
        for i in range(F2 - 1):
            adj_init[i, i + 1] = 1.0    # Connect to next channel
            adj_init[i + 1, i] = 1.0    # Connect to previous channel

        self.adj_scale = nn.Parameter(torch.ones(1))  # Learnable global scaling
        self.register_buffer('adj_structure', adj_init)  # Fixed structure

        # GCAT modules for graph attention
        self.gcat_mod_weight = nn.Linear(self.temporal_len, self.gcat_channels * self.gcat_heads, bias=False)
        self.gcat_mod_val = nn.Linear(self.temporal_len, self.gcat_channels * self.gcat_heads, bias=False)

        # Attention modules (one per head)
        self.gcat_mod_attn_src = nn.ModuleList([
            nn.Linear(self.gcat_channels, 1, bias=False) for _ in range(self.gcat_heads)
        ])
        self.gcat_mod_attn_dst = nn.ModuleList([
            nn.Linear(self.gcat_channels, 1, bias=False) for _ in range(self.gcat_heads)
        ])

        # GCAT layer
        self.gcat = GCAT(
            channels=self.gcat_channels,
            attn_heads=self.gcat_heads,
            concat_heads=False,
            add_self_loops=True,
            attn_dropout_rate=dropout * 0.5,  # Reduced dropout
            activation=nn.ELU(),
            use_bias=False,
            return_attn_coef=False
        )

        # Set GCAT modules
        self.gcat.set_modules(
            self.gcat_mod_weight,
            self.gcat_mod_val,
            self.gcat_mod_attn_src,
            self.gcat_mod_attn_dst
        )

        # Weighted addition for combining original and GCAT features
        self.gcat_weighted_add = WeightedAdd(
            input_size=2,
            weight_initializer='ones',
            weight_constraint=MinMaxValue(min_value=0.0, max_value=1.0)
        )

        # ============================================================
        # Global Convolutional Adaptive Pooling (GCAP) Module
        # ============================================================
        # Adaptive pooling across channels (implicitly done via graph attention)

        # ============================================================
        # Global Temporal Convolution (GTC) Module
        # ============================================================
        # Input: (batch, F2, temporal_len)
        # Conv1D in TensorFlow with channels_last
        # PyTorch Conv1d expects (batch, features, temporal_len)

        self.gtc_pool1 = nn.AvgPool1d(2)
        self.gtc_conv = nn.Conv1d(F2, F2,
                                 kernel_size=3, padding=1, bias=False)
        self.gtc_bn = nn.BatchNorm1d(F2)
        self.gtc_activation = nn.ELU()
        self.gtc_pool2 = nn.AvgPool1d(2)

        # Temporal length after gtc
        self.temporal_len_gtc = self.temporal_len // 4

        # ============================================================
        # Temporal Context Enhancement (TCE) Module
        # ============================================================
        # Positional encoding
        self.tce_pos_encoder = PositionalEncoding(
            trainable_scale=True,
            scale_constraint=MinMaxValue(min_value=0.0, max_value=1.0)
        )

        # Multi-head attention
        self.tce_mha = nn.MultiheadAttention(
            embed_dim=F2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.tce_mha_dropout = nn.Dropout(dropout)

        # Temporal convolution (residual)
        # Conv1D: (batch, features, time)
        self.tce_conv_1 = nn.Conv1d(F2, F2,
                                    kernel_size=3, padding=1, bias=False)
        self.tce_conv_bn = nn.BatchNorm1d(F2)
        self.tce_conv_activation = nn.ELU()
        self.tce_conv_2 = nn.Conv1d(F2, F2,
                                    kernel_size=3, padding=1, bias=False)

        # ============================================================
        # Classification Module
        # ============================================================
        classifier_input_dim = F2 * self.temporal_len_gtc
        self.classifier = nn.Linear(classifier_input_dim, n_classes)

        # Optional feature adapter to match expected feature dimension
        if feature_dim is not None and feature_dim != classifier_input_dim:
            self.feature_adapter = nn.Linear(classifier_input_dim, feature_dim)
        else:
            self.feature_adapter = None

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, 1, Chans, Samples) or (batch, Chans, Samples)

        Returns:
            output: (batch, n_classes)
        """
        # Ensure 4D input: (batch, 1, Chans, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        # ============================================================
        # CTC Module
        # ============================================================
        out = self.ctc_conv(x)  # (batch, F1, Chans, Samples)
        out = self.ctc_bn(out)

        # ============================================================
        # LSFE Module
        # ============================================================
        out = self.lsfe_depthwise(out)  # (batch, F2, 1, Samples)
        out = self.lsfe_bn(out)
        out = self.lsfe_activation(out)
        out = self.lsfe_pool(out)  # (batch, F2, 1, Samples//4)
        out = self.lsfe_dropout(out)

        # ============================================================
        # LTFE Module
        # ============================================================
        out = self.ltfe_depthwise(out)  # (batch, F2, 1, Samples//4)
        out = self.ltfe_pointwise(out)
        out = self.ltfe_bn(out)
        out = self.ltfe_activation(out)
        out = self.ltfe_pool(out)  # (batch, F2, 1, Samples//32)
        out = self.ltfe_dropout(out)

        # Reshape for GCAT: (batch, F2, 1, temporal_len) -> (batch, 1, F2, temporal_len)
        out = out.permute(0, 2, 1, 3)  # (batch, 1, F2, temporal_len)

        # ============================================================
        # GCAT Module - Graph Convolutional Attention Transform
        # ============================================================
        out = out.squeeze(1)  # (batch, F2, temporal_len)
        out0 = out  # Save original features for residual connection

        if not self.disable_gcat:
            # Prepare adjacency matrix (graph structure)
            # Fixed structure with learnable global scaling
            adj = torch.sigmoid(self.adj_scale) * self.adj_structure

            # Apply GCAT: Graph attention with message passing
            # Treats F2 feature channels as graph nodes
            # adj defines which channels can communicate
            out_gcat = self.gcat([out, adj])  # Returns (batch, F2, heads, channels)

            # Average across attention heads
            out_gcat = out_gcat.mean(dim=2)  # (batch, F2, channels)

            # Handle dimension mismatch if needed
            if out_gcat.shape[-1] != out0.shape[-1]:
                if out_gcat.shape[-1] < out0.shape[-1]:
                    padding = out0.shape[-1] - out_gcat.shape[-1]
                    out_gcat = F.pad(out_gcat, (0, padding))
                else:
                    out_gcat = out_gcat[:, :, :out0.shape[-1]]

            # Weighted combination: learnable mix of original and graph-enhanced features
            out = self.gcat_weighted_add([out0, out_gcat])  # (batch, F2, temporal_len)
        # else: skip GCAT, use out0 directly (already assigned to out)

        # ============================================================
        # GCAP Module - Global pooling (done via graph attention)
        # ============================================================

        # ============================================================
        # GTC Module
        # ============================================================
        # Input: (batch, features, temporal_len)
        out = self.gtc_pool1(out)  # (batch, F2, temporal_len//2)
        out = self.gtc_conv(out)
        out = self.gtc_bn(out)
        out = self.gtc_activation(out)
        out = self.gtc_pool2(out)  # (batch, F2, temporal_len//4)

        # ============================================================
        # TCE Module
        # ============================================================
        # Transpose for attention: (batch, features, time) -> (batch, time, features)
        out = out.permute(0, 2, 1)  # (batch, temporal_len_gtc, F2)

        if not self.disable_tce:
            # Positional encoding
            out = self.tce_pos_encoder(out)

            # Multi-head attention with residual
            out0 = out
            out, _ = self.tce_mha(out, out, out)
            out = self.tce_mha_dropout(out)
            out = out + out0  # Residual connection

            # Temporal convolution with residual
            out0 = out
            # Transpose for Conv1d: (batch, time, features) -> (batch, features, time)
            out = out.permute(0, 2, 1)
            out = self.tce_conv_1(out)
            out = self.tce_conv_bn(out)
            out = self.tce_conv_activation(out)
            out = self.tce_conv_2(out)
            # Transpose back and add residual
            out = out.permute(0, 2, 1)
            out = out + out0
        # else: skip TCE, keep out as is

        # ============================================================
        # Classification Module
        # ============================================================
        # Flatten
        features = out.reshape(batch_size, -1)

        # Apply feature adapter if provided
        if self.feature_adapter is not None:
            features_adapted = self.feature_adapter(features)
        else:
            features_adapted = features

        logits = self.classifier(features)

        # Return (adapted_features, logits) to match MVCNet interface
        return features_adapted, logits
