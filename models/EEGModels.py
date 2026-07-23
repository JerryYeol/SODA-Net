"""
EEG Classification Models
- EEGNet
- ShallowConvNet
- DeepConvNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces

    Reference:
    Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based
    brain-computer interfaces. Journal of Neural Engineering.
    """
    def __init__(self, n_classes, Chans=22, Samples=1001, kernLenght=125, F1=4, D=2, F2=8,
                 dropoutRate=0.25, norm_rate=0.25):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernLenght), padding=(0, kernLenght // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise Convolution (Spatial filtering)
        self.conv2 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)

        # Block 2: Separable Convolution
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Calculate feature dimension after convolutions
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self):
        """Calculate the flattened feature dimension"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = F.elu(x)
            x = self.pooling1(x)
            x = self.dropout1(x)
            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = F.elu(x)
            x = self.pooling2(x)
            x = self.dropout2(x)
            return x.numel()

    def forward(self, x):
        """
        Args:
            x: [batch, 1, channels, time]
        Returns:
            features: [batch, feature_dim]
        """
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        return x


class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet from Schirrmeister et al. 2017

    Reference:
    Schirrmeister et al. (2017). Deep learning with convolutional neural networks
    for EEG decoding and visualization. Human Brain Mapping.
    """
    def __init__(self, n_classes, Chans=22, Samples=1001, dropoutRate=0.5):
        super(ShallowConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        # Temporal convolution
        self.conv_temporal = nn.Conv2d(1, 40, (1, 25), bias=True)

        # Spatial convolution
        self.conv_spatial = nn.Conv2d(40, 40, (Chans, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(40)

        # Pooling
        self.pooling = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropoutRate)

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self):
        """Calculate the flattened feature dimension"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)
            x = self.conv_temporal(x)
            x = self.conv_spatial(x)
            x = self.batchnorm(x)
            x = self.square(x)
            x = self.pooling(x)
            x = self.log(x)
            x = self.dropout(x)
            return x.numel()

    def square(self, x):
        return torch.square(x)

    def log(self, x):
        return torch.log(torch.clamp(x, min=1e-6))

    def forward(self, x):
        """
        Args:
            x: [batch, 1, channels, time]
        Returns:
            features: [batch, feature_dim]
        """
        x = self.conv_temporal(x)
        x = self.conv_spatial(x)
        x = self.batchnorm(x)
        x = self.square(x)
        x = self.pooling(x)
        x = self.log(x)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        return x


class DeepConvNet(nn.Module):
    """
    Deep ConvNet from Schirrmeister et al. 2017

    Reference:
    Schirrmeister et al. (2017). Deep learning with convolutional neural networks
    for EEG decoding and visualization. Human Brain Mapping.
    """
    def __init__(self, n_classes, Chans=22, Samples=1001, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples

        # Block 1
        self.conv1 = nn.Conv2d(1, 25, (1, 10), bias=True)
        self.conv2 = nn.Conv2d(25, 25, (Chans, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.pooling1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(dropoutRate)

        # Block 2
        self.conv3 = nn.Conv2d(25, 50, (1, 10), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.pooling2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Block 3
        self.conv4 = nn.Conv2d(50, 100, (1, 10), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.pooling3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout3 = nn.Dropout(dropoutRate)

        # Block 4
        self.conv5 = nn.Conv2d(100, 200, (1, 10), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.pooling4 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.dropout4 = nn.Dropout(dropoutRate)

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self):
        """Calculate the flattened feature dimension"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)

            # Block 1
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.batchnorm1(x)
            x = F.elu(x)
            x = self.pooling1(x)
            x = self.dropout1(x)

            # Block 2
            x = self.conv3(x)
            x = self.batchnorm2(x)
            x = F.elu(x)
            x = self.pooling2(x)
            x = self.dropout2(x)

            # Block 3
            x = self.conv4(x)
            x = self.batchnorm3(x)
            x = F.elu(x)
            x = self.pooling3(x)
            x = self.dropout3(x)

            # Block 4
            x = self.conv5(x)
            x = self.batchnorm4(x)
            x = F.elu(x)
            x = self.pooling4(x)
            x = self.dropout4(x)

            return x.numel()

    def forward(self, x):
        """
        Args:
            x: [batch, 1, channels, time]
        Returns:
            features: [batch, feature_dim]
        """
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        # Block 4
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = self.pooling4(x)
        x = self.dropout4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        return x



class FBCNet(nn.Module):
    """
    Filter Bank Convolutional Neural Network (FBCNet)

    Reference:
    Mane et al. (2021). FBCNet: A Multi-view Convolutional Neural Network for
    Brain-Computer Interface. arXiv:2104.01233.
    """
    def __init__(self, n_classes, Chans=22, Samples=1001, nBands=9, dropoutRate=0.5):
        super(FBCNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.nBands = nBands
        self.dropoutRate = dropoutRate

        # Filter bank (multiple frequency bands)
        self.filter_banks = nn.ModuleList()
        for i in range(nBands):
            # Each filter bank is a temporal convolution
            self.filter_banks.append(
                nn.Conv2d(1, 1, (1, 65), padding=(0, 32), bias=False)
            )

        # Spatial convolution for each band
        self.spatial_conv = nn.ModuleList()
        for i in range(nBands):
            self.spatial_conv.append(
                nn.Conv2d(1, 32, (Chans, 1), bias=False)
            )

        # Batch normalization
        self.batchnorm = nn.ModuleList()
        for i in range(nBands):
            self.batchnorm.append(nn.BatchNorm2d(32))

        # Pooling
        self.pooling = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropoutRate)

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self):
        """Calculate the flattened feature dimension"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)
            features = []

            for i in range(self.nBands):
                xi = self.filter_banks[i](x)
                xi = self.spatial_conv[i](xi)
                xi = self.batchnorm[i](xi)
                xi = torch.square(xi)
                xi = self.pooling(xi)
                xi = torch.log(torch.clamp(xi, min=1e-6))
                xi = self.dropout(xi)
                features.append(xi)

            # Concatenate all bands
            x = torch.cat(features, dim=1)
            return x.numel()

    def forward(self, x):
        """
        Args:
            x: [batch, 1, channels, time]
        Returns:
            features: [batch, feature_dim]
        """
        features = []

        for i in range(self.nBands):
            # Apply filter bank
            xi = self.filter_banks[i](x)
            # Spatial convolution
            xi = self.spatial_conv[i](xi)
            # Batch norm
            xi = self.batchnorm[i](xi)
            # Square activation
            xi = torch.square(xi)
            # Pooling
            xi = self.pooling(xi)
            # Log activation
            xi = torch.log(torch.clamp(xi, min=1e-6))
            # Dropout
            xi = self.dropout(xi)

            features.append(xi)

        # Concatenate features from all filter banks
        x = torch.cat(features, dim=1)

        # Flatten
        x = x.view(x.size(0), -1)

        return x


class ADFCNN(nn.Module):
    """
    Attention-based Deep Feature Convolutional Neural Network (ADFCNN)

    Reference:
    Li et al. (2020). A novel neural network model based on cerebral
    hemispheric asymmetry for EEG emotion recognition.
    """
    def __init__(self, n_classes, Chans=22, Samples=1001, dropoutRate=0.5):
        super(ADFCNN, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples

        # Temporal convolution
        self.conv1 = nn.Conv2d(1, 32, (1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(32)

        # Spatial convolution
        self.conv2 = nn.Conv2d(32, 32, (Chans, 1))
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling1 = nn.AvgPool2d((1, 4))

        # Attention mechanism (use adaptive pooling to match size)
        self.attention_conv = nn.Conv2d(32, 32, (1, 16), padding=0)
        self.attention_bn = nn.BatchNorm2d(32)

        # Deep feature extraction
        self.conv3 = nn.Conv2d(32, 64, (1, 16), padding=(0, 8))
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.pooling2 = nn.AvgPool2d((1, 8))

        self.dropout = nn.Dropout(dropoutRate)

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = F.elu(x)

            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = F.elu(x)
            x = self.pooling1(x)

            # Attention with adaptive size matching
            att = self.attention_conv(x)
            att = self.attention_bn(att)
            if att.size() != x.size():
                att = F.adaptive_avg_pool2d(att, (x.size(2), x.size(3)))
            att = torch.sigmoid(att)
            x = x * att

            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = F.elu(x)
            x = self.pooling2(x)
            x = self.dropout(x)

            return x.numel()

    def forward(self, x):
        # Temporal convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)

        # Spatial convolution
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)

        # Attention mechanism with adaptive size matching
        att = self.attention_conv(x)
        att = self.attention_bn(att)
        # Resize attention map to match x size
        if att.size() != x.size():
            att = F.adaptive_avg_pool2d(att, (x.size(2), x.size(3)))
        att = torch.sigmoid(att)
        x = x * att

        # Deep feature extraction
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        return x


class SlimSeiz(nn.Module):
    """
    SlimSeiz: Lightweight seizure detection network

    A lightweight and efficient CNN for EEG classification.
    """
    def __init__(self, n_classes, Chans=22, Samples=1001, dropoutRate=0.3):
        super(SlimSeiz, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples

        # Lightweight temporal convolution
        self.conv1 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16))
        self.batchnorm1 = nn.BatchNorm2d(16)

        # Depthwise spatial convolution
        self.conv2 = nn.Conv2d(16, 16, (Chans, 1), groups=16)
        self.batchnorm2 = nn.BatchNorm2d(16)

        # Pointwise convolution
        self.conv3 = nn.Conv2d(16, 32, (1, 1))
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pooling1 = nn.AvgPool2d((1, 4))

        # Second block
        self.conv4 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32)
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, (1, 1))
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.pooling2 = nn.AvgPool2d((1, 8))

        self.dropout = nn.Dropout(dropoutRate)

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)

            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = F.relu(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = F.relu(x)
            x = self.pooling1(x)
            x = self.dropout(x)

            x = self.conv4(x)
            x = self.batchnorm4(x)
            x = F.relu(x)

            x = self.conv5(x)
            x = self.batchnorm5(x)
            x = F.relu(x)
            x = self.pooling2(x)
            x = self.dropout(x)

            return x.numel()

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pooling1(x)
        x = self.dropout(x)

        # Second block
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        return x


class CTNet(nn.Module):
    """
    Convolutional Transformer Network (CTNet)

    Combines convolutional layers with self-attention for EEG classification.
    """
    def __init__(self, n_classes, Chans=22, Samples=1001, dropoutRate=0.4):
        super(CTNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples

        # Convolutional feature extraction
        self.conv1 = nn.Conv2d(1, 32, (1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, (Chans, 1))
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling1 = nn.AvgPool2d((1, 4))

        # Temporal convolution
        self.conv3 = nn.Conv2d(32, 64, (1, 16), padding=(0, 8))
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.pooling2 = nn.AvgPool2d((1, 4))

        # Self-attention mechanism (simplified)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.attention_norm = nn.LayerNorm(64)

        self.pooling3 = nn.AvgPool2d((1, 2))
        self.dropout = nn.Dropout(dropoutRate)

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)

            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = F.relu(x)
            x = self.pooling1(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = F.relu(x)
            x = self.pooling2(x)

            # Attention
            b, c, h, w = x.shape
            x_att = x.squeeze(2).permute(0, 2, 1)  # [B, T, C]
            x_att, _ = self.attention(x_att, x_att, x_att)
            x_att = self.attention_norm(x_att)
            x = x_att.permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, T]

            x = self.pooling3(x)
            x = self.dropout(x)

            return x.numel()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pooling1(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pooling2(x)

        # Self-attention
        b, c, h, w = x.shape
        x_att = x.squeeze(2).permute(0, 2, 1)  # [B, T, C]
        x_att, _ = self.attention(x_att, x_att, x_att)
        x_att = self.attention_norm(x_att)
        x = x_att.permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, T]

        x = self.pooling3(x)
        x = self.dropout(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        return x

