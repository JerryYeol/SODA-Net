"""
PyTorch version of GCAT (Graph Convolutional Attention)
Strictly converted from TensorFlow implementation
Modified version of GAT (Graph Attention Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCAT(nn.Module):
    """
    Graph Convolutional Attention Layer
    Modified version of spektral.layers.GATConv
    """
    def __init__(self,
                 channels,
                 attn_heads=1,
                 concat_heads=True,
                 add_self_loops=True,
                 attn_dropout_rate=0.0,
                 activation=None,
                 use_bias=True,
                 return_attn_coef=False):
        super(GCAT, self).__init__()

        self.channels = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.add_self_loops = add_self_loops
        self.attn_dropout_rate = attn_dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.return_attn_coef = return_attn_coef

        # Attention activation (default LeakyReLU)
        self.attn_actvn = nn.LeakyReLU(0.2)

        # Attention dropout
        self.attn_dropout = nn.Dropout(attn_dropout_rate)

        # These will be set externally (matching TensorFlow's mod_weight, mod_val, etc.)
        self.mod_weight = None
        self.mod_val = None
        self.mod_attn_src = None
        self.mod_attn_dst = None

    def set_modules(self, mod_weight, mod_val, mod_attn_src, mod_attn_dst):
        """
        Set the weight, value, and attention modules externally
        This matches TensorFlow's approach of passing modules to __init__

        Args:
            mod_weight: Linear layer for weight projection (channels * attn_heads)
            mod_val: Linear layer for value projection (channels * attn_heads)
            mod_attn_src: List of Linear layers for source attention (one per head)
            mod_attn_dst: List of Linear layers for destination attention (one per head)
        """
        self.mod_weight = mod_weight
        self.mod_val = mod_val
        self.mod_attn_src = nn.ModuleList(mod_attn_src)
        self.mod_attn_dst = nn.ModuleList(mod_attn_dst)

    def forward(self, inputs):
        """
        Forward pass

        Args:
            inputs: list [feat, adj]
                feat: node features (batch, num_nodes, in_features) or (batch, 1, num_nodes, in_features)
                adj: adjacency matrix (num_nodes, num_nodes)

        Returns:
            output: (batch, num_nodes, channels) if concat_heads=False
                   or (batch, num_nodes, channels*attn_heads) if concat_heads=True
            attn_coef_softmax: attention coefficients (optional)
        """
        feat, adj = inputs

        # Handle 4D input: (batch, 1, num_nodes, in_features) -> (batch, num_nodes, in_features)
        if feat.dim() == 4 and feat.shape[1] == 1:
            feat = feat.squeeze(1)

        batch_size, num_nodes, in_features = feat.shape

        # Add self-loops to adjacency matrix if specified
        if self.add_self_loops:
            # adj: (S, D) -> add identity
            identity = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
            adj = adj + identity
            adj = torch.clamp(adj, 0, 1)  # Ensure binary adjacency

        # Compute features for weight and value
        # feat_weight: (batch, num_nodes, channels*attn_heads)
        feat_weight = self.mod_weight(feat)
        # Reshape to (batch, num_nodes, attn_heads, channels)
        feat_weight = feat_weight.view(batch_size, num_nodes, self.attn_heads, self.channels)

        # feat_val: (batch, num_nodes, channels*attn_heads)
        feat_val = self.mod_val(feat)
        # Reshape to (batch, num_nodes, attn_heads, channels)
        feat_val = feat_val.view(batch_size, num_nodes, self.attn_heads, self.channels)

        # Compute attention coefficients for each head
        attn_src_list = []
        attn_dst_list = []

        for h in range(self.attn_heads):
            # attn_src: (batch, num_nodes, 1)
            attn_src = self.mod_attn_src[h](feat_weight[:, :, h, :])
            attn_src_list.append(attn_src)

            # attn_dst: (batch, num_nodes, 1)
            attn_dst = self.mod_attn_dst[h](feat_weight[:, :, h, :])
            attn_dst_list.append(attn_dst)

        # Stack attention scores
        # (batch, num_nodes, attn_heads)
        attn_src = torch.cat(attn_src_list, dim=-1)
        attn_dst = torch.cat(attn_dst_list, dim=-1)

        # Compute pairwise attention: attn_src + attn_dst^T
        # attn_src: (batch, S, 1, H)
        # attn_dst: (batch, 1, D, H)
        # attn_coef: (batch, S, D, H)
        attn_src = attn_src.unsqueeze(2)  # (batch, S, 1, H)
        attn_dst = attn_dst.unsqueeze(1)  # (batch, 1, D, H)
        attn_coef = attn_src + attn_dst  # (batch, S, D, H)

        # Apply attention activation (LeakyReLU)
        attn_coef = self.attn_actvn(attn_coef)

        # Mask attention with adjacency matrix
        # adj: (S, D) -> mask: (S, D)
        # Where adj == 1.0: mask = 0.0, else: mask = -10e9
        adj_mask = torch.where(adj == 1.0,
                              torch.zeros_like(adj),
                              torch.full_like(adj, -10e9))

        # Add mask: (batch, S, D, H) + (S, D, 1)
        adj_mask = adj_mask.unsqueeze(-1)  # (S, D, 1)
        attn_coef = attn_coef + adj_mask

        # Apply softmax over source nodes (dim=1)
        # For each destination node D, sum over source nodes S = 1
        attn_coef_softmax = F.softmax(attn_coef, dim=1)

        # Apply dropout to attention coefficients
        attn_coef_drop = self.attn_dropout(attn_coef_softmax)

        # Message passing: aggregate features weighted by attention
        # attn_coef_drop: (batch, S, D, H)
        # feat_val: (batch, S, H, F)
        # output: (batch, D, H, F)

        # Rearrange feat_val: (batch, num_nodes, attn_heads, channels) -> (batch, S, H, F)
        feat_val = feat_val.permute(0, 1, 2, 3)  # Already in correct shape

        # Einsum: "...SDH, ...SHF -> ...DHF"
        # (batch, S, D, H) x (batch, S, H, F) -> (batch, D, H, F)
        feat_out = torch.einsum("bSDH,bSHF->bDHF", attn_coef_drop, feat_val)

        if self.return_attn_coef:
            return feat_out, attn_coef_softmax
        else:
            return feat_out
