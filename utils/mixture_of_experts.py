"""
Mixture of Experts (MoE) for EEG Classification
多专家混合模型

核心思想：
- 使用多个不同的backbone作为专家（EEGNet, IFNet, Conformer等）
- 每个专家学习不同的特征表示
- 通过门控网络动态选择和组合专家
- 针对不同样本，不同专家的贡献度不同

优势：
1. 集成多个backbone的优势
2. 动态权重，适应不同样本
3. 提升模型容量和表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """
    门控网络

    输入：原始EEG信号或浅层特征
    输出：每个专家的权重
    """
    def __init__(self, input_dim, num_experts, hidden_dim=128):
        super().__init__()

        self.num_experts = num_experts

        # 简单的MLP门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_experts)
        )

        # 温度参数（用于控制专家选择的集中度）
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] - 输入特征

        Returns:
            weights: [batch, num_experts] - 每个专家的权重（和为1）
        """
        # 计算门控logits
        logits = self.gate(x)

        # Softmax with temperature
        weights = F.softmax(logits / self.temperature, dim=1)

        return weights


class SparseGatingNetwork(nn.Module):
    """
    稀疏门控网络

    只激活top-k个专家，减少计算量
    """
    def __init__(self, input_dim, num_experts, top_k=2, hidden_dim=128):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]

        Returns:
            weights: [batch, num_experts] - 稀疏权重（只有top-k个非零）
        """
        # 计算门控logits
        logits = self.gate(x)

        # Top-k选择
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=1)

        # 稀疏权重
        weights = torch.zeros_like(logits)
        weights.scatter_(1, top_indices, F.softmax(top_logits, dim=1))

        return weights


class MixtureOfExperts(nn.Module):
    """
    多专家混合模型

    包含多个backbone专家和一个门控网络
    """
    def __init__(self,
                 experts_dict,
                 classifiers_dict,
                 gate_input_dim,
                 num_classes,
                 gate_type='dense',  # 'dense' or 'sparse'
                 top_k=2,
                 use_auxiliary_loss=True):
        """
        Args:
            experts_dict: {name: expert_network} - 专家字典
            classifiers_dict: {name: classifier} - 每个专家对应的分类器
            gate_input_dim: 门控网络输入维度
            num_classes: 类别数
            gate_type: 门控类型（dense或sparse）
            top_k: 稀疏门控时激活的专家数量
            use_auxiliary_loss: 是否使用辅助损失（平衡专家使用）
        """
        super().__init__()

        self.num_experts = len(experts_dict)
        self.expert_names = list(experts_dict.keys())
        self.use_auxiliary_loss = use_auxiliary_loss

        # 注册专家
        self.experts = nn.ModuleDict(experts_dict)
        self.classifiers = nn.ModuleDict(classifiers_dict)

        # 门控网络
        if gate_type == 'sparse':
            self.gating = SparseGatingNetwork(gate_input_dim, self.num_experts, top_k)
        else:
            self.gating = GatingNetwork(gate_input_dim, self.num_experts)

        # 特征对齐层：将不同专家的特征映射到统一维度
        # 首先需要获取每个专家的特征维度
        self.feature_alignment = nn.ModuleDict()
        self.unified_feature_dim = 512  # 统一特征维度

        # 为每个专家创建特征对齐层
        for expert_name in self.expert_names:
            # 延迟初始化，在第一次forward时根据实际特征维度创建
            self.feature_alignment[expert_name] = None

    def forward(self, x, return_expert_outputs=False):
        """
        Args:
            x: [batch, channels, time] - 输入EEG信号
            return_expert_outputs: 是否返回每个专家的输出

        Returns:
            result: dict - 包含output, gate_weights等
        """
        batch_size = x.shape[0]

        # 1. 计算门控输入（使用全局平均池化）
        # [batch, channels, time] -> [batch, channels]
        gate_input = x.mean(dim=-1)
        gate_input = gate_input.view(batch_size, -1)

        # 2. 计算门控权重
        gate_weights = self.gating(gate_input)  # [batch, num_experts]

        # 3. 每个专家前向传播
        expert_features = []
        expert_logits = []

        for i, expert_name in enumerate(self.expert_names):
            # 准备输入：不同专家可能需要不同的输入格式
            # IFNet需要: [batch, channels, time]
            # EEGNet/ShallowCNN/FBCNet需要: [batch, 1, channels, time]
            if 'EEGNet' in expert_name or 'Shallow' in expert_name or 'FBCNet' in expert_name:
                # 添加1维度给EEGNet/ShallowCNN/FBCNet
                expert_input = x.unsqueeze(1)  # [batch, 1, channels, time]
            else:
                # IFNet使用原始输入
                expert_input = x  # [batch, channels, time]

            # 专家提取特征
            expert_output = self.experts[expert_name](expert_input)

            # 处理不同的返回格式
            if isinstance(expert_output, tuple):
                features = expert_output[0]
            else:
                features = expert_output

            # Flatten features if needed
            if len(features.shape) > 2:
                features = features.view(features.shape[0], -1)

            # 特征对齐：将不同维度的特征映射到统一维度
            if self.feature_alignment[expert_name] is None:
                # 延迟初始化特征对齐层
                feature_dim = features.shape[1]
                if feature_dim != self.unified_feature_dim:
                    self.feature_alignment[expert_name] = nn.Linear(feature_dim, self.unified_feature_dim)
                    if features.is_cuda:
                        self.feature_alignment[expert_name] = self.feature_alignment[expert_name].cuda()
                else:
                    # 如果特征维度已经匹配，使用恒等映射
                    self.feature_alignment[expert_name] = nn.Identity()

            # 应用特征对齐
            aligned_features = self.feature_alignment[expert_name](features)

            # 专家分类
            classifier_output = self.classifiers[expert_name](aligned_features)

            # FC_xy返回 (features, logits)，我们需要logits
            if isinstance(classifier_output, tuple) and len(classifier_output) == 2:
                # 第二个是logits
                logits = classifier_output[1]
            else:
                logits = classifier_output

            # Debug: 打印维度（只在第一次）
            if i == 0 and not hasattr(self, '_debug_printed'):
                print(f"Debug MoE: expert={expert_name}, features shape={features.shape}, aligned shape={aligned_features.shape}, logits shape={logits.shape}")
                self._debug_printed = True

            expert_features.append(aligned_features)  # 使用对齐后的特征
            expert_logits.append(logits)

        # Stack所有专家的输出
        # [num_experts, batch, feature_dim]
        expert_features = torch.stack(expert_features, dim=0)
        # [num_experts, batch, num_classes]
        expert_logits = torch.stack(expert_logits, dim=0)

        # 4. 加权融合
        # [batch, num_experts, 1]
        gate_weights_expanded = gate_weights.unsqueeze(2)

        # 加权求和logits
        # [num_experts, batch, num_classes] -> [batch, num_experts, num_classes]
        expert_logits = expert_logits.permute(1, 0, 2)
        # [batch, num_experts, num_classes] * [batch, num_experts, 1] -> [batch, num_experts, num_classes]
        weighted_logits = expert_logits * gate_weights_expanded
        # [batch, num_classes]
        final_output = weighted_logits.sum(dim=1)

        # 5. 准备返回值
        result = {
            'output': final_output,
            'gate_weights': gate_weights,
        }

        if return_expert_outputs:
            result['expert_logits'] = expert_logits  # [batch, num_experts, num_classes]
            result['expert_features'] = expert_features  # [num_experts, batch, feature_dim]

        return result

    def compute_auxiliary_loss(self, gate_weights):
        """
        辅助损失：鼓励专家使用平衡

        使用负载平衡损失（Load Balancing Loss）
        希望每个专家被平均使用

        Args:
            gate_weights: [batch, num_experts]

        Returns:
            aux_loss: 标量
        """
        if not self.use_auxiliary_loss:
            return torch.tensor(0.0).to(gate_weights.device)

        # 每个专家的平均权重
        mean_weights = gate_weights.mean(dim=0)  # [num_experts]

        # 理想情况：每个专家权重 = 1/num_experts
        target_weight = 1.0 / self.num_experts

        # L2损失
        aux_loss = F.mse_loss(mean_weights, torch.ones_like(mean_weights) * target_weight)

        return aux_loss

    def get_expert_usage_stats(self, gate_weights):
        """
        获取专家使用统计

        Args:
            gate_weights: [batch, num_experts]

        Returns:
            stats: dict - 统计信息
        """
        mean_weights = gate_weights.mean(dim=0).detach().cpu()

        stats = {}
        for i, name in enumerate(self.expert_names):
            stats[name] = mean_weights[i].item()

        return stats


def build_moe_model(args):
    """
    构建MoE模型 - 支持不同的backbone作为专家

    Args:
        args: 参数

    Returns:
        moe_model: MixtureOfExperts实例
    """
    from utils.network import backbone_net_ifnet
    # 导入其他backbone
    from models.EEGModels import EEGNet as EEGNet_Model, ShallowConvNet, FBCNet as FBCNet_Model
    from models.FC import FC_xy

    experts_dict = {}
    classifiers_dict = {}

    # 统一特征维度
    unified_feature_dim = 512

    # 根据args.moe_experts选择专家
    if not hasattr(args, 'moe_experts'):
        args.moe_experts = ['IFNet', 'EEGNet', 'ShallowCNN']

    expert_counter = {}  # 记录每种专家的数量

    for i, expert_name in enumerate(args.moe_experts):
        # 为同一类型的专家生成唯一名称
        if expert_name not in expert_counter:
            expert_counter[expert_name] = 0
        expert_counter[expert_name] += 1

        # 如果是第一个该类型的专家，使用原名；否则加后缀
        if expert_counter[expert_name] == 1:
            unique_name = expert_name
        else:
            unique_name = f"{expert_name}_{expert_counter[expert_name]}"

        try:
            if expert_name == 'IFNet':
                netF, netC = backbone_net_ifnet(args, return_type='xy')
                experts_dict[unique_name] = netF
                # IFNet的特征维度是512，直接使用原分类器
                classifiers_dict[unique_name] = netC

            elif expert_name == 'EEGNet':
                # 创建EEGNet
                netF = EEGNet_Model(
                    n_classes=args.class_num,
                    Chans=args.chn,
                    Samples=args.time_sample_num,
                    kernLenght=int(args.sample_rate // 2),
                    F1=args.F1,
                    D=args.D,
                    F2=args.F2,
                    dropoutRate=args.dropoutRate
                )
                # 使用统一的特征维度创建分类器（特征对齐后）
                netC = FC_xy(unified_feature_dim, args.class_num)
                experts_dict[unique_name] = netF
                classifiers_dict[unique_name] = netC

            elif expert_name == 'ShallowCNN' or expert_name == 'shallow':
                # 创建ShallowConvNet
                netF = ShallowConvNet(
                    n_classes=args.class_num,
                    Chans=args.chn,
                    Samples=args.time_sample_num,
                    dropoutRate=args.dropoutRate
                )
                # 使用统一的特征维度创建分类器（特征对齐后）
                netC = FC_xy(unified_feature_dim, args.class_num)
                experts_dict[unique_name] = netF
                classifiers_dict[unique_name] = netC

            elif expert_name == 'FBCNet':
                # 创建FBCNet
                nBands = 9 if not hasattr(args, 'nBands') else args.nBands
                netF = FBCNet_Model(
                    n_classes=args.class_num,
                    Chans=args.chn,
                    Samples=args.time_sample_num,
                    nBands=nBands,
                    dropoutRate=args.dropoutRate
                )
                # 使用统一的特征维度创建分类器（特征对齐后）
                netC = FC_xy(unified_feature_dim, args.class_num)
                experts_dict[unique_name] = netF
                classifiers_dict[unique_name] = netC

            else:
                print(f"Warning: Expert {expert_name} not supported, skipping...")
                continue

        except Exception as e:
            print(f"Error creating expert {expert_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping this expert...")
            continue

    # 确保至少有一个专家
    if len(experts_dict) == 0:
        raise ValueError("No valid experts specified!")

    print(f"Built MoE with {len(experts_dict)} experts: {list(experts_dict.keys())}")

    # 门控网络输入维度（EEG通道数）
    gate_input_dim = args.chn

    # 构建MoE模型
    moe_model = MixtureOfExperts(
        experts_dict=experts_dict,
        classifiers_dict=classifiers_dict,
        gate_input_dim=gate_input_dim,
        num_classes=args.class_num,
        gate_type=args.moe_gate_type if hasattr(args, 'moe_gate_type') else 'dense',
        top_k=args.moe_top_k if hasattr(args, 'moe_top_k') else 2,
        use_auxiliary_loss=args.moe_use_aux_loss if hasattr(args, 'moe_use_aux_loss') else True
    )

    return moe_model
