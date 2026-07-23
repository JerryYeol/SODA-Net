"""
Hierarchical Multi-Granularity Contrastive Learning
实现多层次对比学习：通道级、时间级、样本级、受试者级
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelContrastiveLoss(nn.Module):
    """
    Level 1: 通道级对比学习
    对比不同通道组（脑区）的表征
    """
    def __init__(self, temperature=0.2, channel_groups=None):
        super().__init__()
        self.temperature = temperature
        self.channel_groups = channel_groups  # 通道分组信息

    def forward(self, features, channel_dim=1):
        """
        Args:
            features: [batch, channels, time] 或 [batch, channels, features]
        """
        batch_size = features.shape[0]
        n_channels = features.shape[channel_dim]

        # 如果没有指定通道分组，自动分成前、中、后三组
        if self.channel_groups is None:
            group_size = n_channels // 3
            self.channel_groups = [
                list(range(0, group_size)),                    # 前部通道
                list(range(group_size, 2*group_size)),         # 中部通道
                list(range(2*group_size, n_channels))          # 后部通道
            ]

        total_loss = 0
        num_pairs = 0

        # 对每个通道组提取特征
        group_features = []
        for group in self.channel_groups:
            if channel_dim == 1:
                group_feat = features[:, group, :].mean(dim=1)  # [batch, time] -> [batch, feat_dim]
            else:
                group_feat = features[:, group, :].mean(dim=1)

            # 归一化
            group_feat = F.normalize(group_feat.reshape(batch_size, -1), dim=1)
            group_features.append(group_feat)

        # 计算组间对比损失（同一样本的不同通道组应该相似）
        for i in range(len(group_features)):
            for j in range(i+1, len(group_features)):
                # 正样本：同一样本的不同通道组
                sim_matrix = torch.matmul(group_features[i], group_features[j].T) / self.temperature

                # 对角线是正样本对
                labels = torch.arange(batch_size).cuda() if features.is_cuda else torch.arange(batch_size)
                loss = F.cross_entropy(sim_matrix, labels)
                total_loss += loss
                num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else total_loss


class TemporalContrastiveLoss(nn.Module):
    """
    Level 2: 时间级对比学习
    对比不同时间窗口的表征
    """
    def __init__(self, temperature=0.2, n_segments=3):
        super().__init__()
        self.temperature = temperature
        self.n_segments = n_segments

    def forward(self, features, time_dim=-1):
        """
        Args:
            features: [batch, channels, time] 或 [batch, time, features]
        """
        batch_size = features.shape[0]
        time_length = features.shape[time_dim]

        # 将时间序列分成n_segments段
        segment_length = time_length // self.n_segments

        segment_features = []
        for i in range(self.n_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length if i < self.n_segments - 1 else time_length

            if time_dim == -1:
                segment = features[..., start_idx:end_idx]
            else:
                segment = features[:, start_idx:end_idx, :]

            # 对时间段内的特征进行平均池化
            segment_feat = segment.mean(dim=time_dim)

            # 归一化
            segment_feat = F.normalize(segment_feat.reshape(batch_size, -1), dim=1)
            segment_features.append(segment_feat)

        total_loss = 0
        num_pairs = 0

        # 时间连续性对比：相邻时间段应该更相似
        for i in range(len(segment_features) - 1):
            # 正样本：同一样本的相邻时间段
            sim_pos = torch.matmul(segment_features[i], segment_features[i+1].T) / self.temperature

            # 负样本：不同样本的时间段
            labels = torch.arange(batch_size).cuda() if features.is_cuda else torch.arange(batch_size)
            loss = F.cross_entropy(sim_pos, labels)
            total_loss += loss
            num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else total_loss


class SubjectContrastiveLoss(nn.Module):
    """
    Level 4: 受试者级对比学习
    学习域不变特征，对比不同受试者的原型
    """
    def __init__(self, temperature=0.2, n_classes=2):
        super().__init__()
        self.temperature = temperature
        self.n_classes = n_classes
        self.prototypes = {}  # 存储每个受试者每个类别的原型

    def update_prototypes(self, features, labels, subject_id):
        """
        更新受试者的类别原型
        Args:
            features: [batch, feature_dim]
            labels: [batch]
            subject_id: int
        """
        features = features.detach()

        if subject_id not in self.prototypes:
            self.prototypes[subject_id] = {}

        for c in range(self.n_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_features = features[mask]
                # 更新原型（指数移动平均）
                new_prototype = class_features.mean(dim=0)

                if c in self.prototypes[subject_id]:
                    # EMA更新
                    self.prototypes[subject_id][c] = 0.9 * self.prototypes[subject_id][c] + 0.1 * new_prototype
                else:
                    self.prototypes[subject_id][c] = new_prototype

    def forward(self, features, labels, subject_ids=None):
        """
        Args:
            features: [batch, feature_dim]
            labels: [batch]
            subject_ids: [batch] - 每个样本的受试者ID
        """
        if len(self.prototypes) < 2:
            # 需要至少2个受试者才能计算对比损失
            return torch.tensor(0.0).to(features.device)

        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)

        # 计算与所有原型的相似度
        all_prototypes = []
        prototype_labels = []

        for subj_id, subj_protos in self.prototypes.items():
            for class_id, proto in subj_protos.items():
                all_prototypes.append(F.normalize(proto.unsqueeze(0).to(features.device), dim=1))
                prototype_labels.append(class_id)

        if len(all_prototypes) == 0:
            return torch.tensor(0.0).to(features.device)

        all_prototypes = torch.cat(all_prototypes, dim=0)  # [n_prototypes, feature_dim]
        prototype_labels = torch.tensor(prototype_labels).to(features.device)

        # 计算特征与原型的相似度
        sim_matrix = torch.matmul(features, all_prototypes.T) / self.temperature  # [batch, n_prototypes]

        # 对比损失：同类别的原型作为正样本（跨受试者）
        loss = 0
        for i in range(batch_size):
            # 找到与当前样本同类别的所有原型（包括不同受试者）
            pos_mask = (prototype_labels == labels[i])

            if pos_mask.sum() > 0:
                # InfoNCE loss
                pos_sim = sim_matrix[i][pos_mask]
                all_sim = sim_matrix[i]

                # log(exp(pos) / sum(exp(all)))
                loss += -torch.log(torch.exp(pos_sim).sum() / torch.exp(all_sim).sum() + 1e-8)

        return loss / batch_size


class HierarchicalContrastiveLoss(nn.Module):
    """
    整合所有层次的对比学习损失
    """
    def __init__(self,
                 temperature=0.2,
                 channel_groups=None,
                 n_time_segments=3,
                 n_classes=2,
                 lambda_channel=1.0,
                 lambda_temporal=1.0,
                 lambda_subject=1.0):
        super().__init__()

        self.channel_contrastive = ChannelContrastiveLoss(temperature, channel_groups)
        self.temporal_contrastive = TemporalContrastiveLoss(temperature, n_time_segments)
        self.subject_contrastive = SubjectContrastiveLoss(temperature, n_classes)

        self.lambda_channel = lambda_channel
        self.lambda_temporal = lambda_temporal
        self.lambda_subject = lambda_subject

    def forward(self, raw_features, global_features, labels, subject_id=None):
        """
        Args:
            raw_features: [batch, channels, time] - 原始空间-时间特征
            global_features: [batch, feature_dim] - 全局特征向量
            labels: [batch] - 类别标签
            subject_id: int - 当前批次的受试者ID（LOSO中同一批次来自同一受试者）
        Returns:
            dict: 包含各层次损失的字典
        """
        losses = {}

        # Level 1: 通道级对比
        if self.lambda_channel > 0:
            loss_channel = self.channel_contrastive(raw_features)
            losses['channel'] = loss_channel
        else:
            losses['channel'] = torch.tensor(0.0)

        # Level 2: 时间级对比
        if self.lambda_temporal > 0:
            loss_temporal = self.temporal_contrastive(raw_features)
            losses['temporal'] = loss_temporal
        else:
            losses['temporal'] = torch.tensor(0.0)

        # Level 4: 受试者级对比
        if self.lambda_subject > 0 and subject_id is not None:
            # 更新原型
            self.subject_contrastive.update_prototypes(global_features, labels, subject_id)
            # 计算损失
            loss_subject = self.subject_contrastive(global_features, labels)
            losses['subject'] = loss_subject
        else:
            losses['subject'] = torch.tensor(0.0)

        # 总损失
        total_loss = (self.lambda_channel * losses['channel'] +
                     self.lambda_temporal * losses['temporal'] +
                     self.lambda_subject * losses['subject'])

        losses['total'] = total_loss

        return losses


# 辅助函数：根据数据集定义通道分组
def get_channel_groups(data_name, n_channels):
    """
    根据数据集返回脑区通道分组
    """
    if 'BNCI2014001' in data_name or data_name == 'BCICIV2a':
        # BNCI2014001 / BCICIV2a: 22 channels
        # 按照10-20系统大致分为前额、中央、顶枕区
        groups = {
            'frontal': [0, 1, 2, 3, 4, 5, 6],          # 前额区
            'central': [7, 8, 9, 10, 11, 12, 13, 14],  # 中央区
            'parietal': [15, 16, 17, 18, 19, 20, 21]   # 顶枕区
        }
        return [groups['frontal'], groups['central'], groups['parietal']]

    elif 'Zhou2016' in data_name:
        # Zhou2016: 14 channels
        groups = {
            'frontal': [0, 1, 2, 3, 4],
            'central': [5, 6, 7, 8, 9],
            'parietal': [10, 11, 12, 13]
        }
        return [groups['frontal'], groups['central'], groups['parietal']]

    elif 'MI1' in data_name:
        # MI1-7: 59 channels - 分成更多组
        group_size = n_channels // 5
        return [
            list(range(0, group_size)),
            list(range(group_size, 2*group_size)),
            list(range(2*group_size, 3*group_size)),
            list(range(3*group_size, 4*group_size)),
            list(range(4*group_size, n_channels))
        ]

    else:
        # 默认：均匀分成3组
        group_size = n_channels // 3
        return [
            list(range(0, group_size)),
            list(range(group_size, 2*group_size)),
            list(range(2*group_size, n_channels))
        ]
