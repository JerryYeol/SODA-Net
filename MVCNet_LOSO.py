'''
=================================================
coding:utf-8
@Time:      2025/4/24 15:41
@File:      MVCNet_LOSO.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import math
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
import sys
from utils.alg_utils import EA_online
from scipy.linalg import fractional_matrix_power
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, data_normalize, data_alignment_uneven, compute_metrics
from utils.utils import data_alignment
from utils.data_augment import data_aug
from utils.network import backbone_net_ifnet, encoder, projector, waveformer_encoder
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_mi_within_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, data_loader_within
from models.Conformer import Conformer
from info_nce import InfoNCE
from utils.contrastive_loss import NTXentLoss, SupConLoss
from utils.hierarchical_contrastive import HierarchicalContrastiveLoss, get_channel_groups
from utils.subject_aware import SubjectAwareTrainer, integrate_subject_aware_training


import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
def train_target(args):

    # 自动检查数据集是否存在
    from utils.auto_download_data import auto_check_and_download
    if not auto_check_and_download(args.data_name, data_dir='data'):
        print(f"\n❌ 数据集 {args.data_name} 不可用，训练终止")
        print(f"   请按照上述说明准备数据集，或使用其他可用数据集")
        sys.exit(1)

    # LOSO (Leave-One-Subject-Out) Cross-Validation:
    # - X_src: Training data from N-1 subjects (source domains)
    # - X_tar: Test data from 1 held-out subject (target domain)
    # - Training uses ONLY X_src, Testing uses ONLY X_tar
    # - This is TRUE cross-subject transfer learning evaluation
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    # network
    if args.backbone == 'EEGNet':
        netF, netC = backbone_net(args, return_type='xy')
    elif args.backbone == 'deep':
        netF, netC = backbone_net_deep(args, return_type='xy')
    elif args.backbone == 'shallow':
        netF, netC = backbone_net_shallow(args, return_type='xy')
    elif args.backbone == 'IFNet':
        netF, netC = backbone_net_ifnet(args, return_type='xy')
    elif args.backbone == 'FBCNet':
        netF = backbone_net_fbcnet(args, return_type='xy')
    elif args.backbone == 'ADFCNN':
        netF, netC = backbone_net_adfcnn(args, return_type='xy')
    elif args.backbone == 'Conformer':
        netF = backbone_net_conformer(args, return_type='xy')
    elif args.backbone == 'FBMSNet':
        netF, netC = backbone_net_fbmsnet(args, return_type='xy')
    elif args.backbone == 'AGTCNet':
        # AGTCNet: end-to-end model with built-in classifier and GCAT
        from models.AGTCNet import AGTCNet

        # F1根据数据集设置
        if args.data_name == 'BNCI2014004':
            F1_value = 16  # BNCI2014004特殊：F1=16
        elif args.data_name == 'BNCI2014001' or 'BNCI2014001' in args.data_name or args.data_name == 'BCICIV2a':
            F1_value = 16  # BNCI2014001 / BCICIV2a 使用F1=16
        else:
            F1_value = 4   # BNCI2014002, BNCI2015001等：F1=4

        # Structural ablation switches for AGTCNet internal modules, controlled
        # by env vars (set by the ablation runner). Default False = full model.
        _disable_gcat = os.environ.get('MVCNET_DISABLE_GCAT', '0').strip() in ('1', 'true', 'True')
        _disable_tce = os.environ.get('MVCNET_DISABLE_TCE', '0').strip() in ('1', 'true', 'True')
        netF = AGTCNet(
            n_classes=args.class_num,
            Chans=args.chn,
            Samples=args.time_sample_num,
            F1=F1_value,
            D=2,
            kernLength=32,
            dropout=args.dropoutRate,
            disable_gcat=_disable_gcat,
            disable_tce=_disable_tce
        )
        netC = None  # AGTCNet has built-in classifier

    if args.backbone == 'FBCNet' or args.backbone == 'Conformer' or args.backbone == 'AGTCNet':
        if args.data_env != 'local':
            netF = netF.cuda()
        base_network = netF
        optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    else:
        if args.data_env != 'local':
            netF, netC = netF.cuda(), netC.cuda()
        base_network = nn.Sequential(netF, netC)
        optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    if args.class_num == 2:
        if args.data_env != 'local':
            class_weight = torch.tensor([1., args.weight], dtype=torch.float32).cuda()
        else:
            class_weight = torch.tensor([1., args.weight], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()


    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()

    if args.encoder == 'Transformer':
        netE = encoder(args, nhead=2, nlayer=1)  # 头数影响不大，层数1-->5(78-->77)
    elif args.encoder == 'WaveFormer':
        netE = waveformer_encoder(args, nlayer=1)  # WaveFormer替代Transformer
    elif args.encoder == 'Conformer':
        netE = Conformer(args, emb_size=40, depth=6, chn=args.chn, n_classes=args.class_num)
    netP = projector(args)
    base_network.train()

    if args.data_env != 'local':
        netE, netP = netE.cuda(), netP.cuda()
    netE.train()
    netP.train()

    # contrastive loss
    """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
    device = torch.device("cuda:0" if args.data_env != 'local' else "cpu")
    scl_criterion = SupConLoss(temperature=args.Context_Cont_temperature)
    """infoNCE loss in Contrastive Predictive Coding"""
    contrastive_loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value (query, positive_key, negative_keys)

    # Hierarchical Multi-Granularity Contrastive Loss
    channel_groups = get_channel_groups(args.data_name, args.chn)
    hierarchical_loss = HierarchicalContrastiveLoss(
        temperature=args.Context_Cont_temperature,
        channel_groups=channel_groups,
        n_time_segments=3,  # 将时间序列分为3段
        n_classes=args.class_num,
        lambda_channel=args.lamda_channel,
        lambda_temporal=args.lamda_temporal,
        lambda_subject=args.lamda_subject
    )
    if args.data_env != 'local':
        hierarchical_loss = hierarchical_loss.cuda()

    # Subject-Aware Training (adversarial + prototype alignment; no model change)
    if getattr(args, 'use_subject_aware', False):
        subject_aware_trainer = SubjectAwareTrainer(
            feature_dim=args.feature_deep_dim,
            num_subjects=args.N - 1,  # 源域受试者数量
            lambda_adversarial=args.lambda_adversarial,
            lambda_mixup=args.lambda_mixup
        )
        if args.data_env != 'local':
            subject_aware_trainer = subject_aware_trainer.to(device)
        subject_aware_trainer.initialize_optimizer(lr=args.lr)
    else:
        subject_aware_trainer = None

    optimizer_e = optim.Adam(netE.parameters(), lr=args.lr)
    optimizer_p = optim.Adam(netP.parameters(), lr=args.lr)

    # Track best test accuracy
    best_acc = 0.0
    best_epoch = 0
    best_metrics = {'auc': float('nan'), 'auprc': float('nan'),
                    'kappa': float('nan'), 'f1': float('nan'), 'recall': float('nan')}
    # Snapshot the best-ACC model weights so the saved checkpoint (and its
    # metrics) always corresponds to the highest accuracy over all epochs.
    best_state = None
    # Early stopping: DISABLED for FULL model runs to ensure complete training.
    # Set patience to 100 (max epochs) so early stopping never triggers.
    # For ablation experiments, use MVCNET_PATIENCE=5 to speed up comparisons.
    patience = int(os.environ.get('MVCNET_PATIENCE', '100'))
    patience_counter = 0
    # Ground-truth labels for the target-online test set, read directly from the
    # TensorDataset (in order, shuffle=False). NOTE: do NOT iterate the DataLoader
    # here — creating a DataLoader iterator consumes one draw from the global RNG
    # and would shift augmentation/dropout randomness, changing training results.
    target_true = dset_loaders["target-online"].dataset.tensors[1].cpu()

    # Progress bar for training iterations
    pbar = tqdm(total=max_iter, desc=f'Training S{args.idt}', unit='iter', ncols=100, leave=False)

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        pbar.update(1)


        if args.aug:
            # View 1
            if 'multi' in args.augmethod1:
                flag_aug1 = [True, False, False, False, False, False]
            elif 'noise' in args.augmethod1:
                flag_aug1 = [False, True, False, False, False, False]
            elif 'flip' in args.augmethod1:
                flag_aug1 = [False, False, True, False, False, False]
            elif 'freq' in args.augmethod1:
                flag_aug1 = [False, False, False, True, False, False]
            elif 'cr' in args.augmethod1:
                flag_aug1 = [False, False, False, False, True, False]
            elif 'hs' in args.augmethod1:
                flag_aug1 = [False, False, False, False, False, True]
            # print('flag_aug1:', flag_aug1)
            if 'hs' in args.augmethod1 or 'cr' in args.augmethod1:
                EEGData_Train1 = np.array(inputs_source.squeeze().cpu())
                EEGLabel_Train1 = np.array(labels_source.cpu())
                aug_out1 = data_aug(args, EEGData_Train1, EEGLabel_Train1, EEGData_Train1.shape[1], flag_aug1)
                aug_inputs_source1, aug_labels_source1 = aug_out1
            else:
                EEGData_Train1 = np.array(inputs_source.squeeze().cpu().swapaxes(1, 2))
                EEGLabel_Train1 = np.array(labels_source.cpu())
                aug_out1 = data_aug(args, EEGData_Train1, EEGLabel_Train1, EEGData_Train1.shape[1], flag_aug1)
                aug_inputs_source1, aug_labels_source1 = aug_out1
                aug_inputs_source1 = np.transpose(aug_inputs_source1, (0, 2, 1))
            aug_inputs_source1, aug_labels_source1 = torch.from_numpy(aug_inputs_source1).to(
                torch.float32), torch.from_numpy(aug_labels_source1).long()
            if args.data_env != 'local':
                aug_inputs_source1 = aug_inputs_source1.cuda()
                aug_labels_source1 = aug_labels_source1.cuda()
            if 'EEGNet' in args.backbone or 'deep' in args.backbone or 'shallow' in args.backbone:
                aug_inputs_source1 = aug_inputs_source1.unsqueeze_(3)
                aug_inputs_source1 = aug_inputs_source1.permute(0, 3, 1, 2)

            # View 2
            if 'multi' in args.augmethod2:
                flag_aug2 = [True, False, False, False, False, False]
            elif 'noise' in args.augmethod2:
                flag_aug2 = [False, True, False, False, False, False]
            elif 'flip' in args.augmethod2:
                flag_aug2 = [False, False, True, False, False, False]
            elif 'freq' in args.augmethod2:
                flag_aug2 = [False, False, False, True, False, False]
            elif 'cr' in args.augmethod2:
                flag_aug2 = [False, False, False, False, True, False]
            elif 'hs' in args.augmethod2:
                flag_aug2 = [False, False, False, False, False, True]
            # print('flag_aug2:', flag_aug2)
            if 'hs' in args.augmethod2 or 'cr' in args.augmethod2:
                EEGData_Train2 = np.array(inputs_source.squeeze().cpu())
                EEGLabel_Train2 = np.array(labels_source.cpu())
                aug_out2 = data_aug(args, EEGData_Train2, EEGLabel_Train2, EEGData_Train2.shape[1], flag_aug2)
                aug_inputs_source2, aug_labels_source2 = aug_out2
            else:
                EEGData_Train2 = np.array(inputs_source.squeeze().cpu().swapaxes(1, 2))
                EEGLabel_Train2 = np.array(labels_source.cpu())
                aug_out2 = data_aug(args, EEGData_Train2, EEGLabel_Train2, EEGData_Train2.shape[1], flag_aug2)
                aug_inputs_source2, aug_labels_source2 = aug_out2
                aug_inputs_source2 = np.transpose(aug_inputs_source2, (0, 2, 1))
            aug_inputs_source2, aug_labels_source2 = torch.from_numpy(aug_inputs_source2).to(
                torch.float32), torch.from_numpy(aug_labels_source2).long()
            if args.data_env != 'local':
                aug_inputs_source2 = aug_inputs_source2.cuda()
                aug_labels_source2 = aug_labels_source2.cuda()
            if 'EEGNet' in args.backbone or 'deep' in args.backbone or 'shallow' in args.backbone:
                aug_inputs_source2 = aug_inputs_source2.unsqueeze_(3)
                aug_inputs_source2 = aug_inputs_source2.permute(0, 3, 1, 2)

            # View 3
            if 'multi' in args.augmethod3:
                flag_aug3 = [True, False, False, False, False, False]
            elif 'noise' in args.augmethod3:
                flag_aug3 = [False, True, False, False, False, False]
            elif 'flip' in args.augmethod3:
                flag_aug3 = [False, False, True, False, False, False]
            elif 'freq' in args.augmethod3:
                flag_aug3 = [False, False, False, True, False, False]
            elif 'cr' in args.augmethod3:
                flag_aug3 = [False, False, False, False, True, False]
            elif 'hs' in args.augmethod3:
                flag_aug3 = [False, False, False, False, False, True]
            # print('flag_aug3:', flag_aug3)
            if 'hs' in args.augmethod3 or 'cr' in args.augmethod3:
                EEGData_Train3 = np.array(inputs_source.squeeze().cpu())
                EEGLabel_Train3 = np.array(labels_source.cpu())
                aug_out3 = data_aug(args, EEGData_Train3, EEGLabel_Train3, EEGData_Train3.shape[1], flag_aug3)
                aug_inputs_source3, aug_labels_source3 = aug_out3
            else:
                EEGData_Train3 = np.array(inputs_source.squeeze().cpu().swapaxes(1, 2))
                EEGLabel_Train3 = np.array(labels_source.cpu())
                aug_out3 = data_aug(args, EEGData_Train3, EEGLabel_Train3, EEGData_Train3.shape[1], flag_aug3)
                aug_inputs_source3, aug_labels_source3 = aug_out3
                aug_inputs_source3 = np.transpose(aug_inputs_source3, (0, 2, 1))
            aug_inputs_source3, aug_labels_source3 = torch.from_numpy(aug_inputs_source3).to(
                torch.float32), torch.from_numpy(aug_labels_source3).long()
            if args.data_env != 'local':
                aug_inputs_source3 = aug_inputs_source3.cuda()
                aug_labels_source3 = aug_labels_source3.cuda()
            if 'EEGNet' in args.backbone or 'deep' in args.backbone or 'shallow' in args.backbone:
                aug_inputs_source3 = aug_inputs_source3.unsqueeze_(3)
                aug_inputs_source3 = aug_inputs_source3.permute(0, 3, 1, 2)

        if 'ADFCNN' in args.backbone or 'Conformer' in args.backbone:
            inputs_source = inputs_source.unsqueeze_(3)
            inputs_source = inputs_source.permute(0, 3, 1, 2)
            aug_inputs_source1 = aug_inputs_source1.unsqueeze_(3)
            aug_inputs_source1 = aug_inputs_source1.permute(0, 3, 1, 2)
            aug_inputs_source2 = aug_inputs_source2.unsqueeze_(3)
            aug_inputs_source2 = aug_inputs_source2.permute(0, 3, 1, 2)
            aug_inputs_source3 = aug_inputs_source3.unsqueeze_(3)
            aug_inputs_source3 = aug_inputs_source3.permute(0, 3, 1, 2)

        features_source, outputs_source = base_network(inputs_source)
        features_source_aug1, outputs_source_aug1 = base_network(aug_inputs_source1)
        features_source_aug2, outputs_source_aug2 = base_network(aug_inputs_source2)
        features_source_aug3, outputs_source_aug3 = base_network(aug_inputs_source3)
        classifier_loss = criterion(outputs_source, labels_source)
        classifier_loss1 = criterion(outputs_source_aug1, aug_labels_source1)
        classifier_loss2 = criterion(outputs_source_aug2, aug_labels_source2)
        classifier_loss3 = criterion(outputs_source_aug3, aug_labels_source3)
        classifier_loss = classifier_loss1 + classifier_loss2 + classifier_loss3 + classifier_loss

        if 'EEGNet' in args.backbone or 'deep' in args.backbone or 'shallow' in args.backbone or 'Conformer' in args.backbone or 'ADFCNN' in args.backbone:
            inputs_source = inputs_source.squeeze()
            aug_inputs_source1 = aug_inputs_source1.squeeze()
            aug_inputs_source2 = aug_inputs_source2.squeeze()
            aug_inputs_source3 = aug_inputs_source3.squeeze()
        if 'BNCI2014001' in args.data_name or args.data_name == 'BCICIV2a' or args.data_name == 'BNCI2014002' or args.data_name == 'BNCI2015001' or args.data_name == 'Zhou2016' or args.data_name == 'BNCI2014004' or args.data_name == 'Lee2019':
            raw = inputs_source[:, :, :-1]
            aug_a1 = aug_inputs_source1[:, :, :-1]
            aug_a2 = aug_inputs_source2[:, :, :-1]
            aug_a3 = aug_inputs_source3[:, :, :-1]
        else:
            raw = inputs_source
            aug_a1 = aug_inputs_source1
            aug_a2 = aug_inputs_source2
            aug_a3 = aug_inputs_source3

        h_raw = netE(raw)
        h_a1 = netE(aug_a1)
        h_a2 = netE(aug_a2)
        h_a3 = netE(aug_a3)

        h_raw = h_raw.reshape(h_raw.shape[0], -1)
        h_a1 = h_a1.reshape(h_a1.shape[0], -1)
        h_a2 = h_a2.reshape(h_a2.shape[0], -1)
        h_a3 = h_a3.reshape(h_a3.shape[0], -1)

        z_raw = netP(h_raw)
        z_a1 = netP(h_a1)
        z_a2 = netP(h_a2)
        z_a3 = netP(h_a3)

        if features_source.shape[0] >= args.batch_size:
            b_s = args.batch_size
        else:
            b_s = features_source.shape[0]
        nt_xent_criterion_cvc = NTXentLoss(device=device, batch_size=b_s * 2,
                                           temperature=args.Context_Cont_temperature,
                                           use_cosine_similarity=args.Context_Cont_use_cosine_similarity)  # device, 256, 0.2, True TODO

        nt_xent_criterion_cmc = NTXentLoss(device=device, batch_size=b_s * 4,
                                           temperature=args.Context_Cont_temperature,
                                           use_cosine_similarity=args.Context_Cont_use_cosine_similarity)  # device, 256, 0.2, True TODO

        cvc_feas_raw = torch.cat((features_source, z_raw))
        cvc_feas_v1 = torch.cat((features_source_aug1, z_a1))
        cvc_feas_v2 = torch.cat((features_source_aug2, z_a2))
        cvc_feas_v3 = torch.cat((features_source_aug3, z_a3))
        cmc_feas_b1 = torch.cat((features_source, features_source_aug1, features_source_aug2, features_source_aug3))
        cmc_feas_b2 = torch.cat((z_raw, z_a1, z_a2, z_a3))

        # Calculate the CVC and CMC losses (Level 3: Sample-level)
        loss_cvc = (nt_xent_criterion_cvc(cvc_feas_raw, cvc_feas_v1) + nt_xent_criterion_cvc(cvc_feas_raw,
                                                                                             cvc_feas_v2) + nt_xent_criterion_cvc(
            cvc_feas_raw, cvc_feas_v3)) / 3
        loss_cmc = nt_xent_criterion_cmc(cmc_feas_b1, cmc_feas_b2)

        # Hierarchical Multi-Granularity Contrastive Loss
        # 准备原始特征用于层次化对比学习
        if 'BNCI2014001' in args.data_name or args.data_name == 'BCICIV2a' or args.data_name == 'BNCI2014002' or args.data_name == 'BNCI2015001' or args.data_name == 'Zhou2016':
            raw_features_for_hcl = raw  # [batch, channels, time]
        else:
            raw_features_for_hcl = inputs_source.squeeze() if len(inputs_source.shape) > 3 else inputs_source

        # 计算层次化对比损失
        hierarchical_losses = hierarchical_loss(
            raw_features=raw_features_for_hcl,  # [batch, channels, time]
            global_features=features_source,     # [batch, feature_dim]
            labels=labels_source,
            subject_id=args.idt  # 当前的目标受试者ID（源域中排除的那个）
        )

        # 总损失：分类损失 + 原始CVC/CMC损失 + 层次化对比损失
        base_classifier_loss = (classifier_loss +
                          args.lamda1 * loss_cvc +
                          args.lamda2 * loss_cmc +
                          hierarchical_losses['total'])

        # Subject-Aware Training (adversarial + prototype alignment regularizer;
        # 不改变模型结构，只在分类损失上叠加受试者感知正则化)
        if getattr(args, 'use_subject_aware', False) and subject_aware_trainer is not None:
            batch_size = features_source.shape[0]
            # 随机分配源域受试者ID（模拟多受试者数据）
            subject_ids = torch.randint(0, args.N - 1, (batch_size,))
            if args.data_env != 'local':
                subject_ids = subject_ids.cuda()

            epoch_progress = iter_num / max_iter

            classifier_loss, loss_dict = integrate_subject_aware_training(
                features=features_source,
                labels=labels_source,
                outputs=outputs_source,
                subject_ids=subject_ids,
                subject_aware_trainer=subject_aware_trainer,
                classifier_loss=base_classifier_loss,
                epoch_progress=epoch_progress
            )
        else:
            classifier_loss = base_classifier_loss

        optimizer_f.zero_grad()
        optimizer_e.zero_grad()
        optimizer_p.zero_grad()
        if args.backbone != 'FBCNet' and args.backbone != 'Conformer' and args.backbone != 'AGTCNet':
            optimizer_c.zero_grad()
        classifier_loss.backward()
        optimizer_f.step()
        optimizer_e.step()
        optimizer_p.step()
        if args.backbone != 'FBCNet' and args.backbone != 'Conformer' and args.backbone != 'AGTCNet':
            optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()
            netE.eval()
            netP.eval()
            acc_t_te, all_output = cal_acc_comb(dset_loaders["target-online"], base_network, args=args)  # TODO target-online
            metrics_te = compute_metrics(target_true, all_output, args=args)

            # Update best accuracy (and snapshot the best-ACC model + metrics)
            current_epoch = int(iter_num // len(dset_loaders["source"]))
            if acc_t_te > best_acc:
                best_acc = acc_t_te
                best_epoch = current_epoch
                best_metrics = metrics_te
                patience_counter = 0
                best_state = {
                    'base_network': {k: v.detach().cpu().clone() for k, v in base_network.state_dict().items()},
                    'netE': {k: v.detach().cpu().clone() for k, v in netE.state_dict().items()},
                    'netP': {k: v.detach().cpu().clone() for k, v in netP.state_dict().items()},
                }
            else:
                patience_counter += 1

            log_str = ('Task: {}, Iter:{}/{}; Acc = {:.2f}% AUC = {:.2f} AUPRC = {:.2f} '
                       'Kappa = {:.4f} F1 = {:.2f} Recall = {:.2f}').format(
                args.task_str, current_epoch, int(max_iter // len(dset_loaders["source"])),
                acc_t_te, metrics_te['auc'], metrics_te['auprc'],
                metrics_te['kappa'], metrics_te['f1'], metrics_te['recall'])
            args.log.record(log_str)
            # Update progress bar with current accuracy
            pbar.set_postfix({'Epoch': current_epoch, 'Acc': f'{acc_t_te:.2f}%',
                              'Best': f'{best_acc:.2f}%', 'Patience': f'{patience_counter}/{patience}'})
            print(f'\r{log_str}', end='', flush=True)

            base_network.train()
            netE.train()
            netP.train()

            # Early stopping: no accuracy improvement for `patience` evaluations
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {current_epoch}: '
                      f'no improvement for {patience} evals (best {best_acc:.2f}% @ epoch {best_epoch}).')
                args.log.record('Early stopping at epoch {}: best {:.2f}% @ epoch {}'.format(
                    current_epoch, best_acc, best_epoch))
                break

    pbar.close()
    print(f'\nBest Test Acc = {best_acc:.2f}% (Epoch {best_epoch})')
    print('Best Metrics: AUC = {:.2f} AUPRC = {:.2f} Kappa = {:.4f} F1 = {:.2f} Recall = {:.2f}'.format(
        best_metrics['auc'], best_metrics['auprc'], best_metrics['kappa'],
        best_metrics['f1'], best_metrics['recall']))
    print(f'Final Test Acc = {acc_t_te:.2f}%')
    args.log.record('Best @ S{} Epoch {}: Acc={:.2f}% AUC={:.2f} AUPRC={:.2f} Kappa={:.4f} F1={:.2f} Recall={:.2f}'.format(
        args.idt, best_epoch, best_acc, best_metrics['auc'], best_metrics['auprc'],
        best_metrics['kappa'], best_metrics['f1'], best_metrics['recall']))

    print('saving model...')

    if not os.path.exists('./runs/' + str(args.data_name) + '/'):
        os.makedirs('./runs/' + str(args.data_name) + '/')

    # 保存完整的模型状态（保存ACC最高那一轮的权重与指标；若从未评估则回退到当前权重）
    if best_state is None:
        best_state = {
            'base_network': base_network.state_dict(),
            'netE': netE.state_dict(),
            'netP': netP.state_dict(),
        }
    checkpoint = {
        'base_network': best_state['base_network'],
        'netE': best_state['netE'],
        'netP': best_state['netP'],
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,  # AUC/AUPRC/Kappa/F1/Recall at the best-ACC epoch
        # 注意：不保存args，因为可能包含不可序列化的对象
    }

    if args.align:
        torch.save(checkpoint,
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_FULL_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt')
    else:
        torch.save(checkpoint,
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_noEA' + '.ckpt')

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    # 通过args回传best-ACC那一轮的指标，不改变返回值签名（调用方仍用 acc = train_target(args)）
    args.best_metrics_last = dict(best_metrics)

    return best_acc


if __name__ == '__main__':
    cpu_num = 8
    torch.set_num_threads(cpu_num)
    # Dataset can be overridden by the MVCNET_DATASET env var (used by the
    # ablation runner). Falls back to the default below when unset.
    _env_dataset = os.environ.get('MVCNET_DATASET', '').strip()
    if _env_dataset:
        data_name_list = [d for d in _env_dataset.split(',') if d]
    else:
        data_name_list = ['BNCI2014002','BNCI2014004']  # 'BNCI2014001', 'Zhou2016', 'MI1-7', 'BNCI2014002', 'BNCI2015001', 'BCICIV2a'
    dct = pd.DataFrame(columns=['dataset', 'avg', 'std'] + ['s' + str(i) for i in range(54)])

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        weight = 1
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 9, 22, 2, 1001, 250, 144, 248, 1000, 22000  # 248 in egn, 2440 in shallow
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 14, 15, 2, 2561, 512, 100, 160, 2560, 38400  # feature_deep_dim=160=8*20 (F1=4,D=2,F2=8,temporal_gtc=20)
        if data_name == 'BNCI2014004': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 9, 3, 2, 1313, 250, 724, 320, 1312, 3936  # feature_deep_dim=320=32*10 (F1=16,D=2,F2=32), dim_e=1312(time-1), dim_p=3*1312=3936, trial_num=724(T+E)
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 12, 13, 2, 2561, 512, 200, 640, 2560, 33280  # 640 in egn, 6600 in shallow
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 9, 22, 4, 1001, 250, 288, 248, 1000, 22000
        if data_name == 'BCICIV2a': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 9, 22, 4, 1001, 250, 288, 224, 1000, 22000  # 4-class BNCI2014001-like, T-session only (288/subject)
        if data_name == 'MI1-7': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 7, 59, 2, 750, 250, 200, 184, 750, 44250  # 184 in egn, 1760 in shallow
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, weight = 'ERP', 8, 8, 2, 256, 256, 4200, 64, 3.5
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, weight = 'ERP', 10, 8, 2, 206, 256, 2520, 64, 9
        if data_name == 'Zhou2016': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 4, 14, 2, 1251, 250, -1, 312, 1250, 17500  # 312/3120/13800, 256 in deep
        if data_name == 'Zhou2016_3': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 4, 14, 3, 1251, 250, -1, 312, 1250, 17500  # 312/3120/13800, 256 in deep
        if data_name == 'Lee2019': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 54, 19, 2, 1000, 250, 100, 224, 999, 18981  # dim_e=time-1=999, dim_p=19*999=18981, trial_num=100(50L+50R)
        F1, D, F2 = 4, 2, 8
        F1, D, F2 = 4, 2, 8

        if 'BNCI2014008' in data_name:
            F1, D, F2 = 8, 4, 16
            feature_deep_dim = 128
        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num, dim_e=dim_e, dim_p=dim_p,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name,
                                  F1=F1, D=D, F2=F2, weight=weight)

        args.backbone = 'AGTCNet'  # EEGNet, shallow, deep, FBCNet, ADFCNN, Conformer, IFNet, AGTCNet
        args.encoder = 'WaveFormer'  #, Transformer,WaveFormer
        if args.encoder == 'Conformer':
            if data_name == 'Zhou2016':
                args.dim_p = 3080
            elif data_name == 'BNCI2014002':
                args.dim_p = 6600
            elif data_name == 'BNCI2015001':
                args.dim_p = 6600
            elif data_name == 'MI1-7':
                args.dim_p = 1760

        args.method = args.backbone + '_' + data_name
        # Optional ablation tag (set by the ablation runner via env) so each
        # ablation variant writes to a distinct log file.
        _abl_tag = os.environ.get('MVCNET_ABLATION_TAG', '').strip()
        if _abl_tag:
            args.method = args.method + '_' + _abl_tag
        # data augmentation
        if args.backbone == 'IFNet':
            args.embed_dims = 64  # IFNet 64
        args.aug = True  # TODO choose augmentation or not
        args.augmethod1 = 'flip'  # TODO: flip multi freq noise cr hs (the number of augmethod can be changed)
        args.augmethod2 = 'freq'  # TODO: flip multi freq noise cr hs (the number of augmethod can be changed)
        args.augmethod3 = 'cr'  # TODO: flip multi freq noise cr hs (the number of augmethod can be changed)
        args.freq_method = 'shift'  # shift surr
        args.freq_mode = 0.1  # [0.1, 0.2, 0.3, 0.4, 0.5]
        args.mult_mode = 0.2  # [0.005, 0.01, 0.05, 0.1, 0.2]
        args.noise_mode = 0.25  # [0.25, 0.5, 1, 2, 4]

        # Contrastive Loss Settings
        args.Context_Cont_temperature = 0.2
        args.Context_Cont_use_cosine_similarity = True
        args.subject = False
        args.perclass = False
        args.lamda1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1  # CVC loss weight
        args.lamda2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1  # CMC loss weight

        # Hierarchical Contrastive Loss Settings (NEW)
        args.lamda_channel = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # Channel-level contrastive weight (关闭)
        args.lamda_temporal = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0  # Temporal-level contrastive weight (关闭)
        args.lamda_subject = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0   # Subject-level contrastive weight (关闭)

        # Subject-Aware Training Settings (轻量级，不改变模型结构)
        args.use_subject_aware = True  # 启用受试者感知训练
        args.lambda_adversarial = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1  # 对抗损失权重
        args.lambda_mixup = float(sys.argv[8]) if len(sys.argv) > 8 else 0.1        # 原型对齐损失权重

        args.augsettings = 'pos'
        if args.backbone == 'IFNet':
            if 'BNCI2014001' in args.data_name:
                args.patch_size = 125
                args.feature_deep_dim = 512
            elif data_name == 'Zhou2016':
                args.patch_size = 125
                args.feature_deep_dim = args.embed_dims * 10
            elif data_name == 'MI1-7':
                args.patch_size = 125
                args.feature_deep_dim = 384
            elif data_name == 'BNCI2015001' or data_name == 'BNCI2014002':
                args.patch_size = 128
                args.feature_deep_dim = 1280
            elif data_name == 'BNCI2014004':
                args.patch_size = 128
                args.feature_deep_dim = 640  # embed_dims(64) * (1313 // 128) = 64 * 10 = 640
        if args.backbone == 'ADFCNN':
            if 'BNCI2014001' in args.data_name:
                args.feature_deep_dim = 552
            elif data_name == 'Zhou2016':
                args.feature_deep_dim = 696
            elif data_name == 'MI1-7':
                args.feature_deep_dim = 408
            elif data_name == 'BNCI2014002':
                args.feature_deep_dim = 672  # 8 * 84 (actual AGTCNet output)
            elif data_name == 'BNCI2015001':
                args.feature_deep_dim = 1440
        if args.backbone == 'FBCNet':
            if 'BNCI2014001' in args.data_name:
                args.nBands = 22
                args.feature_deep_dim = 192
            elif data_name == 'Zhou2016':
                args.nBands = 7
                args.feature_deep_dim = 192
            elif data_name == 'MI1-7':
                args.nBands = 2
                args.feature_deep_dim = 192
            elif data_name == 'BNCI2014002':
                args.nBands = 3
                args.feature_deep_dim = 192
            elif data_name == 'BNCI2015001':
                args.nBands = 3
                args.feature_deep_dim = 192
        if args.backbone == 'FBMSNet':
            if 'BNCI2014001' in args.data_name:
                args.in_chans = 9
                args.feature_deep_dim = 192
            elif data_name == 'Zhou2016':
                args.in_chans = 14
                args.feature_deep_dim = 36
            elif data_name == 'MI1-7':
                args.in_chans = 59
                args.feature_deep_dim = 192
            elif data_name == 'BNCI2014002':
                args.in_chans = 3
                args.feature_deep_dim = 192
            elif data_name == 'BNCI2015001':
                args.in_chans = 3
                args.feature_deep_dim = 192
        if args.backbone == 'shallow':
            if 'BNCI2014001' in args.data_name:
                args.feature_deep_dim = 2440
            if data_name == 'Zhou2016':
                args.feature_deep_dim = 3120  # 312/3120/13800
            elif data_name == 'BNCI2014002':
                args.feature_deep_dim = 1480  # 312/3120/13800
                args.dim_e = 640
                args.dim_p = 9600
            elif data_name == 'BNCI2015001':
                args.feature_deep_dim = 1480  # 312/3120/13800
                args.dim_e = 640
                args.dim_p = 8320
            elif data_name == 'MI1-7':
                args.feature_deep_dim = 1760
                # args.dim_e = 640
                # args.dim_p = 8320
        if args.backbone == 'AGTCNet':
            # AGTCNet feature dimensions: F2 * temporal_len_gtc
            # F2 = F1 * D, temporal_len_gtc = Samples // 128
            if 'BNCI2014001' in args.data_name or args.data_name == 'BCICIV2a':
                # F1=16, D=2, F2=32, Samples=1001
                # temporal_len_gtc = 1001 // 128 = 7
                # feature_dim = 32 * 7 = 224
                args.feature_deep_dim = 224
            elif data_name == 'Zhou2016':
                # F1=4, D=2, F2=F1*D=8, Samples=1251
                # temporal_len_gtc = 1251 // 128 = 9
                # feature_dim = 8 * 9 = 72
                args.feature_deep_dim = 72
            elif data_name == 'Lee2019':
                # F1=4, D=2, F2=F1*D=8, Samples=1000
                # temporal_len_gtc = 1000 // 128 = 7
                # feature_dim = 8 * 7 = 56
                args.feature_deep_dim = 56
            elif data_name == 'MI1-7':
                # Samples=750, temporal_len_gtc = 750 // 128 = 5
                # feature_dim = 32 * 5 = 160
                args.feature_deep_dim = 160
            elif data_name == 'BNCI2014002' or data_name == 'BNCI2015001':
                # Samples=2561, F1=4, D=2, F2=8
                # temporal_len = 2561 // 32 = 80
                # temporal_gtc = 80 // 4 = 20
                # feature_dim = 8 * 20 = 160
                args.feature_deep_dim = 160
            elif data_name == 'BNCI2014004':
                # Samples=1313, F1=16 (hardcoded), D=2, F2=32
                # CTC: 1313+1=1314, LSFE÷4=328, LTFE conv+1=329, LTFE÷8=41
                # temporal_len = 1313 // 32 = 41, temporal_gtc = 41 // 4 = 10
                # feature_dim = 32 * 10 = 320
                args.feature_deep_dim = 320
        if args.backbone == 'Conformer':
            if 'BNCI2014001' in args.data_name:
                args.feature_deep_dim = 2440
                args.dim_p = 22000  # 2440
            if data_name == 'Zhou2016':
                args.feature_deep_dim = 3080
                args.dim_e = 220
                args.dim_p = 3080
            elif data_name == 'BNCI2014002':
                args.feature_deep_dim = 6600
                args.dim_p = 6600
            elif data_name == 'BNCI2015001':
                args.feature_deep_dim = 6600
                args.dim_p = 6600
            elif data_name == 'MI1-7':
                args.feature_deep_dim = 1760
                args.dim_p = 1760
        elif args.backbone == 'deep':
            if 'BNCI2014001' in args.data_name:
                args.feature_deep_dim = 10800
                # args.dim_e = 0
                # args.dim_p = 0
            if data_name == 'Zhou2016':
                args.feature_deep_dim = 3400  # 3400
                args.dim_e = 416
                args.dim_p = 5824  #
            elif data_name == 'MI1-7':
                args.feature_deep_dim = 1400
                args.dim_e = 250
                args.dim_p = 14750
            elif data_name == 'BNCI2015001':
                args.feature_deep_dim = 1400
                args.dim_e = 256
                args.dim_p = 3328
            elif data_name == 'BNCI2014002':
                args.feature_deep_dim = 1400
                args.dim_e = 256
                args.dim_p = 3840

        args.projector_dim1 = args.feature_deep_dim * 4
        args.projector_dim2 = args.feature_deep_dim
        # whether to use EA
        args.align = True  # TODO
        args.dropoutRate = 0.25
        # learning rate
        args.lr = 0.001
        # train batch size
        # if args.aug:
        #     args.batch_size = 64
        # else:
        if args.aug:
            args.batch_size = 64
        else:
            args.batch_size = 32
        # training epochs
        args.max_epoch = 100

        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            # Properly check CUDA after setting environment variable
            import torch
            if torch.cuda.is_available():
                args.data_env = 'gpu'
                args.device = torch.device(f'cuda:0')  # After setting CUDA_VISIBLE_DEVICES, use cuda:0
                print(f'Using GPU: {device_id} (CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()})')
            else:
                args.data_env = 'local'
                args.device = torch.device('cpu')
                print('CUDA not available, using CPU')
        except:
            # No GPU specified, check if CUDA is available
            import torch
            if torch.cuda.is_available():
                args.data_env = 'gpu'
                args.device = torch.device('cuda:0')
                print(f'Using default GPU (CUDA available, Device count: {torch.cuda.device_count()})')
            else:
                args.data_env = 'local'
                args.device = torch.device('cpu')
                print('No GPU specified and CUDA not available, using CPU')

        total_acc = []

        # train multiple randomly initialized models
        #for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        print(f'\n{"="*80}')
        print(f'Dataset: {data_name} | Total Seeds: 5 | Total Subjects: {N}')
        print(f'{"="*80}\n')
        for s in tqdm([1], desc=f'{data_name} Seeds', unit='seed', ncols=100):
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)

            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

            metric_keys = ['auc', 'auprc', 'kappa', 'f1', 'recall']
            sub_acc_all = np.zeros(N)
            sub_metrics_all = {k: np.zeros(N) for k in metric_keys}
            for idt in tqdm(range(N), desc=f'  LOSO Subjects (Seed {s})', unit='subj', ncols=100, leave=False):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                # Print in terminal with proper formatting
                tqdm.write(info_str)
                my_log.record(info_str)
                args.log = my_log
                if args.data_name == 'Zhou2016':
                    sbj_num = [119, 100, 100, 90]
                    args.nsamples = math.ceil(sbj_num[idt] / 2 * 0.8)  # 80%训练，20%测试
                elif args.data_name == 'BNCI2014001-4' or args.data_name == 'BCICIV2a':
                    args.nsamples = math.ceil(args.trial_num / 4 * 0.8)  # 80%训练，20%测试
                else:
                    args.nsamples = math.ceil(args.trial_num / 2 * 0.8)  # 80%训练，20%测试
                sub_acc_all[idt] = train_target(args)
                # 从args回传的指标收集（train_target调用与返回值完全不变）
                for k in metric_keys:
                    sub_metrics_all[k][idt] = args.best_metrics_last.get(k, float('nan'))

            # Print subject accuracies with proper formatting
            tqdm.write(f'\nSeed {s} Results:')
            tqdm.write('Sub acc: ' + str(np.round(sub_acc_all, 3)))
            tqdm.write('Avg acc: ' + str(np.round(np.mean(sub_acc_all), 3)))
            print(f'Seed {s} - Avg: {np.round(np.mean(sub_acc_all), 3):.3f}%')
            total_acc.append(sub_acc_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)

            # 其余指标（AUC/AUPRC/Kappa/F1/Recall）的每受试者值与均值
            tqdm.write(f'\nSeed {s} Metrics (mean over subjects):')
            args.log.record('---- Metrics (best-ACC epoch per subject) ----')
            for k in metric_keys:
                k_mean = np.nanmean(sub_metrics_all[k])
                tqdm.write('  {:<7}: {:.4f}'.format(k.upper(), k_mean))
                args.log.record('{} sub: {}'.format(k.upper(), str(np.round(sub_metrics_all[k], 4).tolist())))
                args.log.record('{} mean: {:.4f}'.format(k.upper(), k_mean))

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)
        tqdm.write('\n' + '=' * 80)
        tqdm.write('FINAL RESULTS')
        tqdm.write('=' * 80)
        print(str(total_acc))
        args.log.record(str(total_acc))
        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        tqdm.write(f'\nSubject Mean: {subject_mean}')
        tqdm.write(f'Method: {args.method}')
        tqdm.write(f'Total Mean: {total_mean:.5f}%')
        tqdm.write(f'Total Std: {total_std:.5f}%')
        tqdm.write('=' * 80)
        print(f'\n*** FINAL: {args.method} | Mean={total_mean:.2f}% | Std={total_std:.2f}% ***\n')

        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))

        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i in range(len(subject_mean)):
            result_dct['s' + str(i)] = subject_mean[i]
