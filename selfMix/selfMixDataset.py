import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


logger = logging.getLogger(__name__)

NUM_CLASSES = {
    'agnews': 4
}

class DataToDataset(Dataset):
    def __init__(self, data):
        self.labels, self.gtlabels, self.texts, self.group = data.values[:, 0], data.values[:, 1],data.values[:, 2], data.values[:, 3]  
        # 添加 sample_id，使用行号作为唯一标识
        self.sample_ids = np.arange(len(self.labels))
        
    def __len__(self):
        return len(self.labels)
            
    def __getitem__(self,index):
        return self.texts[index], self.labels[index], self.gtlabels[index], self.group[index], self.sample_ids[index]
    

def rawload_dataset(data_path, dataset_name):
    extension = data_path.split(".")[-1]
    assert extension == 'csv'
    
    data = pd.read_csv(data_path, header=None)
    
    if dataset_name in NUM_CLASSES:
        num_classes = NUM_CLASSES[dataset_name]
    else:
        num_classes = max(data.values[:, 0]) + 1
        
    logger.info('num_classes is %d', num_classes)
    return DataToDataset(data), num_classes

def load_dataset(data_path, dataset_name):
    extension = data_path.split(".")[-1]
    assert extension == 'csv'
    
    # 加载数据时指定 header=0 以处理列头
    data = pd.read_csv(data_path, header=0)
    
    # 确保标签列和第二列为整数类型
    data.iloc[:, 0] = data.iloc[:, 0].astype(int)  # 将第一列转换为整数类型
    data.iloc[:, 1] = data.iloc[:, 1].astype(int)  # 将第二列转换为整数类型
    
    if dataset_name in NUM_CLASSES:
        num_classes = NUM_CLASSES[dataset_name]
    else:
        num_classes = max(data.iloc[:, 0]) + 1  # 计算类别数
        
    # 计算数据统计信息
    total_samples = len(data)
    class_distribution = data.iloc[:, 0].value_counts().sort_index()
    noise_rate = 1 - (data.iloc[:, 0] == data.iloc[:, 1]).mean() if len(data.columns) > 1 else 0
    
    logger.info(f'数据集统计信息:')
    logger.info(f'- 样本总数: {total_samples}')
    logger.info(f'- 类别数: {num_classes}')
    logger.info(f'- 噪声率: {noise_rate:.2%}')
    logger.info('- 各类别样本分布:')
    for class_idx, count in class_distribution.items():
        logger.info(f'  类别 {class_idx}: {count} 样本 ({count/total_samples:.2%})')
        
    return DataToDataset(data), num_classes
    

class SelfMixDataset(Dataset):
    def __init__(self, data_args, dataset, tokenizer, mode, pred=[], probability=[]): 
        self.data_args = data_args
        self.labels = dataset.labels
        self.gtlabels = dataset.gtlabels
        self.group = dataset.group
        self.texts = dataset.texts
        # 添加 sample_id
        self.sample_ids = dataset.sample_ids
        
       
        
        # 如果概率为空，则默认概率为0.5
        if len(probability) == 0 :
            self.prob = len(self.texts) * [0.5]
            logger.info("概率为空，默认概率为0.5")
        else:
            logger.info("概率不为空，使用给定的概率")
            logger.info(f"概率长度: {len(probability)}")
            self.prob = probability

        self.mode = mode
        self.tokenizer = tokenizer
        # 打印数据长度和结构，用于调试
        logger.info(f"原始数据集长度: {len(dataset.labels)}")
        logger.info(f"处理后的数据集长度:")
        logger.info(f"- texts: {len(self.texts)}")  
        logger.info(f"- labels: {len(self.labels)}")
        logger.info(f"- gtlabels: {len(self.gtlabels)}")
        logger.info(f"- group: {len(self.group)}")
        logger.info(f"- sample_ids: {len(self.sample_ids)}")
        logger.info(f"- prob: {len(self.prob)}")

        if self.mode == "labeled":
            # pred 是一个0-1之间的概率值，pred_idx 是概率值大于阈值的索引
            pred_idx = pred.nonzero()[0]
            self.texts = [self.texts[idx] for idx in pred_idx]
            self.labels = self.labels[pred_idx]
            self.gtlabels = self.gtlabels[pred_idx]
            self.group = self.group[pred_idx]
            self.sample_ids = self.sample_ids[pred_idx]
            self.prob = [probability[idx] for idx in pred_idx]
            self.pred_idx = pred_idx
        elif self.mode == "unlabeled":
            pred_idx = (1 - pred).nonzero()[0]
            self.texts = [self.texts[idx] for idx in pred_idx]
            self.prob = [probability[idx] for idx in pred_idx]
            self.labels = self.labels[pred_idx]
            self.gtlabels = self.gtlabels[pred_idx]
            self.group = self.group[pred_idx]
            self.sample_ids = self.sample_ids[pred_idx]
            self.pred_idx = pred_idx   
                            
    def __len__(self):
        return len(self.texts)
    
    def get_tokenized(self, text):
        tokens = self.tokenizer(text, padding='max_length', truncation=True, 
                               max_length=self.data_args.max_sentence_len, return_tensors='pt')
        for item in tokens:
            tokens[item] = tokens[item].squeeze()
        
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

    def __getitem__(self, index):
        text = self.texts[index]
        input_id, att_mask = self.get_tokenized(text)
        
        # 确保所有返回值都是张量
        label = torch.tensor(self.labels[index], dtype=torch.long)
        gtlabel = torch.tensor(self.gtlabels[index], dtype=torch.long)
        prob_value = torch.tensor(self.prob[index], dtype=torch.float)
        sample_id = torch.tensor(self.sample_ids[index], dtype=torch.long)
        
        if self.mode == 'labeled' or self.mode == 'unlabeled':
            pred_idx_value = torch.tensor(self.pred_idx[index], dtype=torch.long)
        else:
            pred_idx_value = torch.tensor(index, dtype=torch.long)
        
     
        
        return input_id, att_mask, label, prob_value, pred_idx_value, gtlabel, self.group[index], sample_id


class SelfMixData:
    def __init__(self, data_args, datasets, tokenizer):
        self.data_args = data_args
        self.datasets = datasets
        self.tokenizer = tokenizer
    
    def run(self, mode, pred=[], prob=[]):
        if mode == "all":
            all_dataset = SelfMixDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="all")
                
            all_loader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.data_args.batch_size,
                shuffle=True,
                num_workers=2)          
            return all_loader

        if mode == "train":
            labeled_dataset = SelfMixDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="labeled", 
                pred=pred, probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.data_args.batch_size_mix,
                shuffle=True,
                num_workers=2)   
            
            unlabeled_dataset = SelfMixDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="unlabeled", 
                pred=pred,probability=prob)              
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.data_args.batch_size_mix,
                shuffle=True,
                num_workers=2)     
            return labeled_trainloader, unlabeled_trainloader
