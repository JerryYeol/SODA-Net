#!/usr/bin/env python3
"""
自动数据集管理和下载
当训练时检测到数据集不存在，自动尝试下载或给出获取方式
"""

import os
import sys
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm


class DatasetDownloader:
    """数据集下载管理器"""

    DATASETS_INFO = {
        'BNCI2014001': {
            'description': 'BCI Competition IV Dataset 2a (9 subjects, 4 classes MI)',
            'download_method': 'mne',
            'auto_downloadable': True,
        },
        'BCICIV2a': {
            'description': 'BCI Competition IV Dataset 2a (9 subjects, 4 classes MI), preprocessed like BNCI2014001',
            'download_method': 'manual',
            'auto_downloadable': False,
        },
        'BNCI2014002': {
            'description': 'BCI Competition IV Dataset 2b (9 subjects, 2 classes MI)',
            'download_method': 'mne',
            'auto_downloadable': True,
        },
        'BNCI2015001': {
            'description': 'BCI Competition 2015 Dataset 1 (12 subjects)',
            'download_method': 'mne',
            'auto_downloadable': True,
        },
        'Zhou2016': {
            'description': 'Zhou 2016 Motor Imagery Dataset (4 subjects, 3 classes)',
            'download_method': 'manual',
            'auto_downloadable': False,
            'paper': 'https://doi.org/10.1038/sdata.2016.10',
        },
        'MI1': {'description': 'MI Dataset 1', 'download_method': 'manual', 'auto_downloadable': False},
        'MI2': {'description': 'MI Dataset 2', 'download_method': 'manual', 'auto_downloadable': False},
        'MI3': {'description': 'MI Dataset 3', 'download_method': 'manual', 'auto_downloadable': False},
        'MI4': {'description': 'MI Dataset 4', 'download_method': 'manual', 'auto_downloadable': False},
        'MI5': {'description': 'MI Dataset 5', 'download_method': 'manual', 'auto_downloadable': False},
        'MI6': {'description': 'MI Dataset 6', 'download_method': 'manual', 'auto_downloadable': False},
        'MI7': {'description': 'MI Dataset 7', 'download_method': 'manual', 'auto_downloadable': False},
    }

    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def check_and_download(self, dataset_name):
        """检查数据集，如果不存在则尝试下载"""
        dataset_path = self.data_dir / dataset_name

        # 检查数据集是否已存在
        if self._is_dataset_complete(dataset_name):
            print(f"✓ 数据集 {dataset_name} 已存在且完整")
            return True

        print(f"✗ 数据集 {dataset_name} 不存在或不完整")

        # 尝试自动下载
        info = self.DATASETS_INFO.get(dataset_name, {})

        if info.get('auto_downloadable'):
            print(f"→ 尝试自动下载 {dataset_name}...")
            return self._auto_download(dataset_name)
        else:
            print(f"→ {dataset_name} 需要手动下载")
            self._print_manual_instructions(dataset_name)
            return False

    def _is_dataset_complete(self, dataset_name):
        """检查数据集文件是否完整"""
        dataset_path = self.data_dir / dataset_name

        if not dataset_path.exists():
            return False

        # 检查必需文件
        required_files = ['X.npy', 'y.npy']
        for file in required_files:
            if not (dataset_path / file).exists():
                # 检查是否有 labels.npy 可以作为 y.npy
                if file == 'y.npy' and (dataset_path / 'labels.npy').exists():
                    # 创建符号链接
                    os.symlink('labels.npy', dataset_path / 'y.npy')
                    continue
                return False

        return True

    def _auto_download(self, dataset_name):
        """自动下载数据集"""
        info = self.DATASETS_INFO[dataset_name]
        method = info.get('download_method')

        if method == 'mne':
            return self._download_via_mne(dataset_name)
        else:
            print(f"✗ 不支持的下载方法: {method}")
            return False

    def _download_via_mne(self, dataset_name):
        """通过MNE-Python下载数据集"""
        try:
            import mne
            from mne.datasets import eegbci
        except ImportError:
            print("✗ 需要安装 mne 库来自动下载数据集")
            print("  安装命令: pip install mne")
            return False

        print(f"→ 使用MNE下载 {dataset_name}...")

        try:
            if dataset_name == 'BNCI2014001':
                return self._download_bnci2014001_mne()
            elif dataset_name == 'BNCI2014002':
                return self._download_bnci2014002_mne()
            elif dataset_name == 'BNCI2015001':
                return self._download_bnci2015001_mne()
            else:
                print(f"✗ 暂不支持通过MNE下载 {dataset_name}")
                return False
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return False

    def _download_bnci2014001_mne(self):
        """下载BNCI2014001数据集"""
        print("→ 下载BNCI2014001可能需要较长时间...")
        print("  建议手动下载预处理好的数据")
        return False

    def _download_bnci2014002_mne(self):
        """下载BNCI2014002数据集"""
        print("→ BNCI2014002需要手动下载")
        self._print_manual_instructions('BNCI2014002')
        return False

    def _download_bnci2015001_mne(self):
        """下载BNCI2015001数据集"""
        print("→ BNCI2015001需要手动下载")
        self._print_manual_instructions('BNCI2015001')
        return False

    def _print_manual_instructions(self, dataset_name):
        """打印手动下载说明"""
        print("\n" + "=" * 70)
        print(f"📥 {dataset_name} 手动下载指南")
        print("=" * 70)

        info = self.DATASETS_INFO.get(dataset_name, {})
        print(f"\n描述: {info.get('description', 'N/A')}")

        if dataset_name == 'BNCI2014001':
            print("""
方法1: 从GitHub下载预处理数据 (推荐)
--------------------------------------
一些研究者分享了预处理好的numpy格式数据:

1. 搜索: "BNCI2014001 numpy" 或 "BCI Competition IV 2a preprocessed"
2. 下载 X.npy 和 y.npy (或 labels.npy)
3. 放到: data/BNCI2014001/

方法2: 从官方网站下载原始数据
--------------------------------------
1. 访问: http://www.bbci.de/competition/iv/#dataset2a
2. 下载所有受试者的GDF文件 (A01-A09)
3. 使用MNE或自定义脚本预处理
4. 转换为numpy格式

数据格式要求:
  X.npy: shape (N, 22, 1001) - EEG信号
  y.npy: shape (N,) - 标签 (0-3, 4个类别)
            """)

        elif dataset_name == 'BNCI2014002':
            print("""
1. 访问: http://www.bbci.de/competition/iv/#dataset2b
2. 下载数据文件
3. 预处理为numpy格式:
   X.npy: shape (N, channels, time_points)
   y.npy: shape (N,) - 标签
4. 放到: data/BNCI2014002/
            """)

        elif dataset_name == 'Zhou2016':
            print("""
Zhou 2016数据集:
1. 论文: https://doi.org/10.1038/sdata.2016.10
2. 数据: https://figshare.com/articles/dataset/...
3. 下载并预处理为numpy格式
4. 放到: data/Zhou2016/
            """)

        else:
            print(f"""
请联系数据集提供者或查看相关论文获取数据。

数据应放置在:
  data/{dataset_name}/X.npy  - EEG信号数据
  data/{dataset_name}/y.npy  - 标签数据
            """)

        print("=" * 70)


def auto_check_and_download(dataset_name, data_dir='data'):
    """
    自动检查并下载数据集
    在训练脚本中调用此函数
    """
    downloader = DatasetDownloader(data_dir)

    if downloader.check_and_download(dataset_name):
        print(f"✓ 数据集 {dataset_name} 准备就绪")
        return True
    else:
        print(f"✗ 数据集 {dataset_name} 不可用")
        print(f"\n请按照上述说明手动下载数据集，或选择其他可用数据集")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='EEG数据集下载工具')
    parser.add_argument('dataset', type=str, help='数据集名称')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录')

    args = parser.parse_args()

    success = auto_check_and_download(args.dataset, args.data_dir)
    sys.exit(0 if success else 1)
