"""
Subject-Aware Training Strategy for LOSO
受试者感知训练策略

创新点：
1. 在训练时显式追踪每个源受试者的表征
2. 通过受试者级别的对抗训练增强域不变性
3. 使用Mixup在受试者间进行插值，学习平滑的跨域表征
4. 不改变原始模型结构、损失函数和数据增强

理论基础：
- 让模型意识到"受试者"这个概念
- 学习受试者不变的特征表征
- 通过对抗训练和Mixup增强泛化能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SubjectDiscriminator(nn.Module):
    """
    受试者判别器
    用于对抗训练，强制特征提取器学习受试者不变的表征
    """
    def __init__(self, feature_dim, num_subjects):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_subjects)
        )

    def forward(self, features):
        """
        预测特征来自哪个受试者
        Args:
            features: [batch, feature_dim]
        Returns:
            subject_logits: [batch, num_subjects]
        """
        return self.discriminator(features)


class GradientReversalLayer(torch.autograd.Function):
    """
    梯度反转层 (Gradient Reversal Layer)
    前向传播：直接传递
    反向传播：梯度乘以-lambda（反转梯度）

    用途：让特征提取器学习判别器无法区分的特征
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class SubjectAwareTrainer:
    """
    受试者感知训练器

    在不改变原始模型的情况下，通过训练策略增强跨受试者泛化
    """
    def __init__(self, feature_dim, num_subjects, lambda_adversarial=0.1, lambda_mixup=0.5):
        """
        Args:
            feature_dim: 特征维度
            num_subjects: 源受试者数量（N-1）
            lambda_adversarial: 对抗损失权重
            lambda_mixup: Mixup损失权重
        """
        self.feature_dim = feature_dim
        self.num_subjects = num_subjects
        self.lambda_adversarial = lambda_adversarial
        self.lambda_mixup = lambda_mixup

        # 受试者判别器
        self.subject_discriminator = SubjectDiscriminator(feature_dim, num_subjects)

        # 判别器优化器
        self.discriminator_optimizer = None  # 稍后初始化

        # 受试者原型（用于对齐）
        self.subject_prototypes = {}  # {subject_id: {class_id: prototype}}

    def to(self, device):
        """移动到设备"""
        self.subject_discriminator = self.subject_discriminator.to(device)
        return self

    def initialize_optimizer(self, lr=0.001):
        """初始化判别器优化器"""
        self.discriminator_optimizer = torch.optim.Adam(
            self.subject_discriminator.parameters(),
            lr=lr
        )

    def update_prototypes(self, features, labels, subject_ids, momentum=0.9):
        """
        更新受试者-类别原型

        Args:
            features: [batch, feature_dim]
            labels: [batch]
            subject_ids: [batch] - 每个样本的受试者ID
        """
        features = features.detach()

        unique_subjects = torch.unique(subject_ids)

        for subj_id in unique_subjects.cpu().numpy():
            if subj_id not in self.subject_prototypes:
                self.subject_prototypes[subj_id] = {}

            # 找到该受试者的样本
            subj_mask = (subject_ids == subj_id)
            subj_features = features[subj_mask]
            subj_labels = labels[subj_mask]

            # 对每个类别更新原型
            unique_labels = torch.unique(subj_labels)
            for class_id in unique_labels.cpu().numpy():
                class_mask = (subj_labels == class_id)
                class_features = subj_features[class_mask]

                if class_features.shape[0] > 0:
                    new_prototype = class_features.mean(dim=0)

                    # 指数移动平均
                    if class_id in self.subject_prototypes[subj_id]:
                        old_prototype = self.subject_prototypes[subj_id][class_id]
                        self.subject_prototypes[subj_id][class_id] = \
                            momentum * old_prototype + (1 - momentum) * new_prototype
                    else:
                        self.subject_prototypes[subj_id][class_id] = new_prototype

    def compute_adversarial_loss(self, features, subject_ids, lambda_grl=1.0):
        """
        计算对抗损失

        目标：让特征提取器学习判别器无法区分受试者的特征

        Args:
            features: [batch, feature_dim]
            subject_ids: [batch]
            lambda_grl: 梯度反转层的lambda

        Returns:
            adversarial_loss: 对抗损失
        """
        # 梯度反转
        reversed_features = GradientReversalLayer.apply(features, lambda_grl)

        # 判别器预测
        subject_logits = self.subject_discriminator(reversed_features)

        # 判别器损失（特征提取器视角：希望判别器预测错误）
        adversarial_loss = F.cross_entropy(subject_logits, subject_ids)

        return adversarial_loss

    def train_discriminator(self, features, subject_ids):
        """
        训练判别器（独立步骤）

        Args:
            features: [batch, feature_dim]
            subject_ids: [batch]

        Returns:
            discriminator_loss: 判别器损失
            discriminator_acc: 判别器准确率
        """
        # 停止特征的梯度（只训练判别器）
        features = features.detach()

        # 判别器预测
        subject_logits = self.subject_discriminator(features)

        # 判别器损失（希望正确分类受试者）
        discriminator_loss = F.cross_entropy(subject_logits, subject_ids)

        # 更新判别器
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # 计算准确率
        pred_subjects = subject_logits.argmax(dim=1)
        discriminator_acc = (pred_subjects == subject_ids).float().mean().item()

        return discriminator_loss.item(), discriminator_acc

    def compute_prototype_alignment_loss(self, features, labels):
        """
        计算原型对齐损失

        目标：让同类别的不同受试者的特征接近跨受试者的平均原型

        Args:
            features: [batch, feature_dim]
            labels: [batch]

        Returns:
            alignment_loss: 原型对齐损失
        """
        if len(self.subject_prototypes) < 2:
            return torch.tensor(0.0).to(features.device)

        # 计算每个类别的跨受试者平均原型
        global_prototypes = {}
        for subj_id, subj_protos in self.subject_prototypes.items():
            for class_id, proto in subj_protos.items():
                if class_id not in global_prototypes:
                    global_prototypes[class_id] = []
                global_prototypes[class_id].append(proto)

        for class_id in global_prototypes:
            global_prototypes[class_id] = torch.stack(global_prototypes[class_id]).mean(dim=0)

        # 计算特征与对应类别全局原型的距离
        alignment_loss = 0
        count = 0

        for i in range(features.shape[0]):
            class_id = labels[i].item()
            if class_id in global_prototypes:
                # L2距离
                dist = F.mse_loss(features[i], global_prototypes[class_id].to(features.device))
                alignment_loss += dist
                count += 1

        if count > 0:
            alignment_loss = alignment_loss / count
        else:
            alignment_loss = torch.tensor(0.0).to(features.device)

        return alignment_loss

    def subject_mixup(self, features1, labels1, features2, labels2, alpha=0.5):
        """
        受试者间的Mixup

        在特征空间中混合不同受试者的样本

        Args:
            features1, labels1: 受试者1的特征和标签
            features2, labels2: 受试者2的特征和标签
            alpha: Beta分布参数

        Returns:
            mixed_features: 混合后的特征
            mixed_labels_a: 标签a
            mixed_labels_b: 标签b
            lambda_: 混合比例
        """
        batch_size = min(features1.shape[0], features2.shape[0])

        if alpha > 0:
            lambda_ = np.random.beta(alpha, alpha)
        else:
            lambda_ = 1.0

        # 随机选择样本
        indices1 = torch.randperm(features1.shape[0])[:batch_size]
        indices2 = torch.randperm(features2.shape[0])[:batch_size]

        # 混合特征
        mixed_features = lambda_ * features1[indices1] + (1 - lambda_) * features2[indices2]

        return mixed_features, labels1[indices1], labels2[indices2], lambda_


def integrate_subject_aware_training(
    features,
    labels,
    outputs,
    subject_ids,
    subject_aware_trainer,
    classifier_loss,
    epoch_progress=0.0
):
    """
    整合受试者感知训练到现有训练流程

    Args:
        features: [batch, feature_dim] - 从backbone提取的特征
        labels: [batch] - 类别标签
        outputs: [batch, num_classes] - 分类器输出
        subject_ids: [batch] - 受试者ID
        subject_aware_trainer: SubjectAwareTrainer实例
        classifier_loss: 原始分类损失
        epoch_progress: 当前epoch进度 (0-1)

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
    """
    # 1. 更新受试者原型
    subject_aware_trainer.update_prototypes(features, labels, subject_ids)

    # 2. 训练判别器（每隔几步训练一次）
    discriminator_loss = 0
    discriminator_acc = 0
    if torch.rand(1).item() < 0.5:  # 50%概率训练判别器
        discriminator_loss, discriminator_acc = subject_aware_trainer.train_discriminator(
            features, subject_ids
        )

    # 3. 计算对抗损失（梯度反转）
    # lambda随训练进度增加（从0到1）
    lambda_grl = 2.0 / (1.0 + np.exp(-10 * epoch_progress)) - 1.0
    adversarial_loss = subject_aware_trainer.compute_adversarial_loss(
        features, subject_ids, lambda_grl
    )

    # 4. 计算原型对齐损失
    alignment_loss = subject_aware_trainer.compute_prototype_alignment_loss(features, labels)

    # 5. 组合损失
    total_loss = (
        classifier_loss +
        subject_aware_trainer.lambda_adversarial * adversarial_loss +
        subject_aware_trainer.lambda_mixup * alignment_loss
    )

    loss_dict = {
        'classifier': classifier_loss.item(),
        'adversarial': adversarial_loss.item(),
        'alignment': alignment_loss.item(),
        'discriminator': discriminator_loss,
        'disc_acc': discriminator_acc
    }

    return total_loss, loss_dict
