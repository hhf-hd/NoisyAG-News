import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

class NoiseDataset(Dataset):
    """
    噪声标签数据集
    """
    def __init__(self, file_path, tokenizer_path, max_length=256):
        """
        初始化数据集
        Args:
            file_path: CSV文件路径
            tokenizer_path: tokenizer路径或名称
            max_length: 最大序列长度
        """
        # 读取数据
        self.df = pd.read_csv(file_path)
        
        self.texts = self.df['text'].values
        self.noisy_labels = self.df['label'].values
        self.clean_labels = self.df['gt'].values
        self.groups = self.df['group'].values
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        
        # 计算数据集统计信息
        self.num_classes = len(np.unique(self.clean_labels))
        self.noise_rate = (self.noisy_labels != self.clean_labels).mean()
        self.group_stats = pd.Series(self.groups).value_counts().to_dict()
        
        # 记录统计信息
        logger.info(f"Dataset loaded from {file_path}")
        logger.info(f"Total samples: {len(self.df)}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Noise rate: {self.noise_rate:.2%}")
        logger.info(f"Group distribution: {self.group_stats}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        noisy_label = self.noisy_labels[idx]
        clean_label = self.clean_labels[idx]
        group = self.groups[idx]

        # tokenize文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'noisy_label': torch.tensor(noisy_label, dtype=torch.long),
            'clean_label': torch.tensor(clean_label, dtype=torch.long),
            'group': group,
            'text': text,
            'sample_id': idx
        }

def create_dataloader(file_path, tokenizer_path, batch_size, max_length=256, shuffle=True):
    """
    创建数据加载器
    Args:
        file_path: CSV文件路径
        tokenizer_path: tokenizer路径或名称
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
    Returns:
        dataloader: DataLoader实例
        dataset: Dataset实例
    """
    dataset = NoiseDataset(file_path, tokenizer_path, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, dataset
