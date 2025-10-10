import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np


logger = logging.getLogger(__name__)

NUM_CLASSES = {
    'trec': 6,
    'imdb': 2,
    'agnews': 4,
    'chn': 2,
}


class DataToDataset(Dataset):
    def __init__(self, data):
        # 添加sample_id列，使用行索引
        self.sample_ids = np.arange(len(data))  # 新增：为每行添加唯一的sample_id
        self.labels = data.values[:, 0]  # label列
        self.true_labels = data.values[:, 1]  # gt列
        self.texts = data.values[:, 2]  # text列
        self.groups = data.values[:, 3]  # group列
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 返回值中添加sample_id
        return self.texts[index], self.labels[index], self.true_labels[index], self.groups[index], self.sample_ids[index]
    

def load_dataset(data_path, dataset_name):
    extension = data_path.split(".")[-1]
    assert extension == 'csv'
    
    # 读取CSV文件，现在CSV有header
    data = pd.read_csv(data_path)
    
    if dataset_name in NUM_CLASSES:
        num_classes = NUM_CLASSES[dataset_name]
    else:
        num_classes = max(data.values[:, 0]) + 1
        
    logger.info('num_classes is %d', num_classes)
    return DataToDataset(data), num_classes
    

class expDecayDataset(Dataset):
    def __init__(self, data_args, dataset, tokenizer, mode, pred=[], probability=[]): 
        self.data_args = data_args
        self.labels = dataset.labels
        self.inputs = dataset.texts
        self.true_labels = dataset.true_labels
        self.groups = dataset.groups
        self.sample_ids = dataset.sample_ids  # 新增：保存sample_ids
        self.mode = mode
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def get_tokenized(self, text):
        tokens = self.tokenizer(text, padding='max_length', truncation=True, 
                                max_length=self.data_args.max_sentence_len, return_tensors='pt')

        for item in tokens:
            tokens[item] = tokens[item].squeeze()
        
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

    def __getitem__(self, index):
        text = self.inputs[index]
        input_id, att_mask = self.get_tokenized(text)
        if self.mode == 'all':
            # 返回值中添加sample_id
            return input_id, att_mask, self.labels[index], self.true_labels[index], self.groups[index], self.sample_ids[index]


class expDecayData:
    def __init__(self, data_args, datasets, tokenizer):
        self.data_args = data_args
        self.datasets = datasets
        self.tokenizer = tokenizer
    
    def run(self, mode):
        if mode == "all":
            all_dataset = expDecayDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="all")
                
            all_loader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.data_args.batch_size,
                shuffle=True,  # 虽然shuffle=True，但是通过sample_id仍然可以追踪到具体样本
                num_workers=0
            )          # num_workers=2
            return all_loader