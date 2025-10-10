import torch
from transformers import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    AdamW,
    AutoTokenizer
)
from sklearn.metrics import accuracy_score
import logging
import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
# 设置日志

logger = logging.getLogger(__name__)

class DualModelDataset(Dataset):
    """
    支持双模型训练的数据集
    """
    def __init__(self, file_path, model_names, max_length=256):
        """
        Args:
            model_names: 要支持的模型名称列表，如['albert', 'bert']
        """
        self.df = pd.read_csv(file_path)
        self.texts = self.df['text'].values
        self.noisy_labels = self.df['label'].values
        self.clean_labels = self.df['gt'].values
        self.groups = self.df['group'].values
        
        # 初始化多个tokenizer
        self.tokenizers = {
            name: AutoTokenizer.from_pretrained(self._get_model_path(name)) 
            for name in model_names
        }
        self.max_length = max_length
        
        # 统计信息
        self.num_classes = len(np.unique(self.clean_labels))
        self.noise_rate = (self.noisy_labels != self.clean_labels).mean()
        
        logger.info(f"Loaded dataset with {len(self)} samples")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Noise rate: {self.noise_rate:.2%}")

    def _get_model_path(self, name):
        """映射模型名称到预训练名称"""
        model_map = {
            'albert': 'albert-base-v2',
            'bert': 'bert-base-uncased'
        }
        return model_map[name]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = {}
        
        # 为每个模型生成编码
        for name, tokenizer in self.tokenizers.items():
            encoding = tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encodings[f'{name}_input_ids'] = encoding['input_ids'].squeeze(0)
            encodings[f'{name}_attention_mask'] = encoding['attention_mask'].squeeze(0)
        
        return {
            **encodings,
            'noisy_label': torch.tensor(self.noisy_labels[idx], dtype=torch.long),
            'clean_label': torch.tensor(self.clean_labels[idx], dtype=torch.long),
            'text': text,
            'group': torch.tensor(self.groups[idx], dtype=torch.long),
            'sample_id': idx
        }

def create_dual_dataloader(file_path, model_names, batch_size=32, max_length=256, shuffle=True):
    """创建双模型数据加载器"""
    dataset = DualModelDataset(file_path, model_names, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    return loader, dataset

class DualModelTrainer:
    def __init__(self, model_names, num_labels, device='cuda', lr=2e-5, forget_rate=0.2):
        self.models = {
            'albert': AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=num_labels).to(device),
            'bert': BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)
        }
        self.optimizers = {
            name: AdamW(model.parameters(), lr=lr)
            for name, model in self.models.items()
        }
       
        self.device = device
        self.best_acc = {name: 0.0 for name in model_names}
        self.forget_rate = forget_rate  # 遗忘率，用于控制每次选择的样本比例
        
    def _compute_loss(self, outputs, labels):
        """计算每个样本的损失"""
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        return criterion(outputs.logits, labels)
    
    def _select_samples(self, losses1, losses2):
        """根据损失值选择可能的干净样本"""
        num_remember = int(len(losses1) * (1 - self.forget_rate))
        
        # 获取损失值较小的样本索引
        _, idx1 = torch.topk(losses1, num_remember, largest=False)
        _, idx2 = torch.topk(losses2, num_remember, largest=False)
        
        return idx1, idx2
    
    
    def warmup(self, data_loader):
        """使用50个batch预热训练"""
        # 两个模型使用不同的batch训练
        batch_count = 0
        logger.info("Warmup training...")
        
        for batch in data_loader:
            
            if batch_count > 100:
                break
            batch_count += 1
            losses = {}
            
            # 准备两个模型的输入
            model_inputs = {
                'albert': {
                    'input_ids': batch['albert_input_ids'].to(self.device),
                    'attention_mask': batch['albert_attention_mask'].to(self.device),
                },
                'bert': {
                    'input_ids': batch['bert_input_ids'].to(self.device),
                    'attention_mask': batch['bert_attention_mask'].to(self.device),
                }
            }
            labels = batch['noisy_label'].to(self.device)
            # 使用选定的样本训练两个模型
            self.models['albert'].train()
            self.models['bert'].train()
            
            if batch_count >= 0 and batch_count <= 50:
            
                # 训练albert模型
                outputs_albert = self.models['albert'](**model_inputs['albert'])
                loss_albert = self._compute_loss(outputs_albert, labels)
                loss_albert = torch.mean(loss_albert)  # 只使用bert选择的样本
                loss_albert.backward()
                self.optimizers['albert'].step()
                self.optimizers['albert'].zero_grad()
                losses['albert'] = loss_albert.item()
            
            if batch_count >= 50 and batch_count <= 100:
            
            
                # 训练bert模型
                outputs_bert = self.models['bert'](**model_inputs['bert'])
                loss_bert = self._compute_loss(outputs_bert, labels)
                loss_bert = torch.mean(loss_bert)  
                loss_bert.backward()
                self.optimizers['bert'].step()
                self.optimizers['bert'].zero_grad()
                losses['bert'] = loss_bert.item()
            
            
        self.evaluate_data(data_loader,type='warmup')
        
        logger.info("Warmup training complete")
        
        
        
        
    def train_step(self, batch):
        """改进的训练步骤"""
        losses = {}
        
        # 准备输入
        model_inputs = {
            'albert': {
                'input_ids': batch['albert_input_ids'].to(self.device),
                'attention_mask': batch['albert_attention_mask'].to(self.device),
            },
            'bert': {
                'input_ids': batch['bert_input_ids'].to(self.device),
                'attention_mask': batch['bert_attention_mask'].to(self.device),
            }
        }
        labels = batch['noisy_label'].to(self.device)
        
        # 计算每个模型的损失
        sample_losses = {}
        with torch.no_grad():
            for name, model in self.models.items():
                model.eval()
                outputs = model(**model_inputs[name])
                sample_losses[name] = self._compute_loss(outputs, labels)
               
        
        # 选择小损失样本
        idx1, idx2 = self._select_samples(sample_losses['albert'], sample_losses['bert'])
        
        # 使用选定的样本训练两个模型
        self.models['albert'].train()
        self.models['bert'].train()
        
        # 训练albert模型（使用bert选择的样本）
        outputs_albert = self.models['albert'](**model_inputs['albert'])
        loss_albert = self._compute_loss(outputs_albert, labels)
        loss_albert = torch.mean(loss_albert[idx2])  # 只使用bert选择的样本
        loss_albert.backward()
        self.optimizers['albert'].step()
        self.optimizers['albert'].zero_grad()
        losses['albert'] = loss_albert.item()
        
        # 训练bert模型（使用albert选择的样本）
        outputs_bert = self.models['bert'](**model_inputs['bert'])
        loss_bert = self._compute_loss(outputs_bert, labels)
        loss_bert = torch.mean(loss_bert[idx1])  # 只使用albert选择的样本
        loss_bert.backward()
        self.optimizers['bert'].step()
        self.optimizers['bert'].zero_grad()
        losses['bert'] = loss_bert.item()
            
        return losses,idx1,idx2
    
    def evaluate_data(self, data_loader,type='val'):
        """改进的评估函数"""
        
        # logger.info(f'evaluate {self.models.keys()}')
        logger.info(f'evaluate {type} ')
        results = {}
        maxAcc = 0
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name} model...")
            model.eval()
            predictions = []
            true_labels = []
            noisy_labels = []
            groups = []
            
            with torch.no_grad():
                batch_num = 0
                for batch in data_loader:
                    batch_num += 1
                   
                    inputs = {
                        'input_ids': batch[f'{name}_input_ids'].to(self.device),
                        'attention_mask': batch[f'{name}_attention_mask'].to(self.device)
                    }
                    
                    clean_labels = batch['clean_label'].numpy()
                    batch_noisy_labels = batch['noisy_label'].numpy()
                    batch_groups = batch['group'].numpy()
                    
                    outputs = model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    
                    predictions.extend(preds)
                    true_labels.extend(clean_labels)
                    noisy_labels.extend(batch_noisy_labels)
                    groups.extend(batch_groups)
                
                # 计算整体准确率
                clean_acc = accuracy_score(true_labels, predictions)
                noisy_acc = accuracy_score(noisy_labels, predictions)
                
                if clean_acc > maxAcc:
                    maxAcc = clean_acc
                
                # 按组别计算准确率
                group_results = {}
                unique_groups = np.unique(groups)
                for group in unique_groups:
                    group_mask = np.array(groups) == group
                    group_clean_acc = accuracy_score(
                        np.array(true_labels)[group_mask],
                        np.array(predictions)[group_mask]
                    )
                    group_noisy_acc = accuracy_score(
                        np.array(noisy_labels)[group_mask],
                        np.array(predictions)[group_mask]
                    )
                    group_results[int(group)] = {
                        'clean_acc': group_clean_acc,
                        'noisy_acc': group_noisy_acc
                    }
                
                results[name] = {
                    'overall_clean_acc': clean_acc,
                    'overall_noisy_acc': noisy_acc,
                    'group_results': group_results
                }
                
                # 输出详细结果
                logger.info(f"\n{name.upper()} Model Results:")
                logger.info(f"{type} Overall Clean Accuracy: {clean_acc:.2%}")
                logger.info(f"{type} Overall Noisy Accuracy: {noisy_acc:.2%}")
                logger.info("\nGroup-wise Results:")
                for group, group_acc in group_results.items():
                    logger.info(f"Group {group}:")
                    logger.info(f"  Clean Accuracy: {group_acc['clean_acc']:.2%}")
                    logger.info(f"  Noisy Accuracy: {group_acc['noisy_acc']:.2%}")
        
       
        logger.info(f'evaluate end')
        logger.info(f'*'*100)
        return results,maxAcc
    
    def _save_model(self, model_name, acc):
        """保存最佳模型"""
        os.makedirs(f"models/{model_name}", exist_ok=True)
        path = f"models/{model_name}/best_acc_{acc:.4f}.pt"
        torch.save(self.models[model_name].state_dict(), path)
        logger.info(f"Saved {model_name} model with acc {acc:.2%}")

def analyze_model_selections(idx1_list, idx2_list, batch_groups, batch_ids):
    """分析两个模型选择的样本
    
    Args:
        idx1_list: albert选择的样本索引列表
        idx2_list: bert选择的样本索引列表
        batch_groups: 批次中的组别信息
        batch_ids: 批次中的样本ID
    
    Returns:
        dict: 包含分析结果的字典
    """
    # 将tensor转换为numpy数组
    idx1 = idx1_list.cpu().numpy()
    idx2 = idx2_list.cpu().numpy()
    
    # 计算共同选择的样本
    common_samples = np.intersect1d(idx1, idx2)
    total_samples = np.union1d(idx1, idx2)
    
    # 计算相似度
    similarity = len(common_samples) / len(total_samples) if len(total_samples) > 0 else 0
    
    # 分析每个组的样本分布
    selected_groups = batch_groups[common_samples].cpu().numpy()
    group_distribution = {}
    unique_groups = np.unique(selected_groups)
    for group in unique_groups:
        group_count = np.sum(selected_groups == group)
        group_distribution[int(group)] = {
            'count': int(group_count),
            'percentage': float(group_count / len(common_samples)) if len(common_samples) > 0 else 0
        }
    
    return {
        'similarity': similarity,
        'common_samples_count': len(common_samples),
        'total_selected_samples': len(total_samples),
        'group_distribution': group_distribution
    }

def train_dual_models(
    train_loader,
    val_loader,
    test_loader,
    num_labels,
    model_names,
    num_epochs=5,
    device='cuda',
    forget_rate=0.2
):
    """改进的双模型Co-Teaching训练"""
    trainer = DualModelTrainer(model_names, num_labels, device, forget_rate=forget_rate)
    logger.info(f'train_dual_models {model_names}')
    
    trainer.warmup(train_loader)
    
    epoch_statistics = []
    
    start_time = time.time()
    
    val_maxAcc = []
    test_maxAcc = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        total_losses = {name: 0.0 for name in model_names}
        progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}', ncols=100)
        
        epoch_idx1_list = []
        epoch_idx2_list = []
        epoch_groups = []
        epoch_sample_ids = []
        
        for batch_idx, batch in enumerate(progress_bar):
            losses, idx1, idx2 = trainer.train_step(batch)
            
            # 收集每个batch的选择信息
            epoch_idx1_list.extend(idx1.cpu().numpy())
            epoch_idx2_list.extend(idx2.cpu().numpy())
            epoch_groups.extend(batch['group'].numpy())
            epoch_sample_ids.extend(batch['sample_id'].numpy())
            
            # 更新损失统计
            for name in model_names:
                total_losses[name] += losses[name]
                
            progress_bar.set_postfix({
                'albert_loss': losses['albert'],
                'bert_loss': losses['bert']
            })
            
            # 每100步验证一次
            if batch_idx % 200 == 0:
                val_results,maxAcc = trainer.evaluate_data(val_loader)
                test_results,maxAcc = trainer.evaluate_data(test_loader,type='test')
                val_maxAcc.append(maxAcc)
                test_maxAcc.append(maxAcc)
                
                # logger.info(f"Step {batch_idx} | Albert Val: {val_results['albert']:.2%} | Bert Val: {val_results['bert']:.2%}")
        
        
        
        # 计算平均损失
        avg_losses = {name: total_losses[name]/len(train_loader) for name in model_names}
        logger.info(f"Epoch {epoch+1} Avg Losses - Albert: {avg_losses['albert']:.4f}, Bert: {avg_losses['bert']:.4f}")
        
        # 分析模型选择的样本
        selection_analysis = analyze_model_selections(
            torch.tensor(epoch_idx1_list),
            torch.tensor(epoch_idx2_list),
            torch.tensor(epoch_groups),
            torch.tensor(epoch_sample_ids)
        )
        
        # 评估阶段
        val_results,ValmaxAcc = trainer.evaluate_data(val_loader)
        test_results,TestmaxAcc = trainer.evaluate_data(test_loader,type='test')
        val_maxAcc.append(ValmaxAcc)
        test_maxAcc.append(TestmaxAcc)
        
        # 记录本epoch的统计信息
        epoch_stat = {
            'epoch': epoch + 1,
            'avg_losses': avg_losses,
            'validation_results': val_results,
            'selection_analysis': selection_analysis
        }
        epoch_statistics.append(epoch_stat)
        
        # 输出详细的统计信息
        logger.info(f"\nEpoch {epoch+1} Statistics:")
        logger.info(f"Average Losses - Albert: {avg_losses['albert']:.4f}, Bert: {avg_losses['bert']:.4f}")
        # logger.info(f"Validation Results - Albert: {val_results['albert']:.2%}, Bert: {val_results['bert']:.2%}")
        logger.info("\nModel Selection Analysis:")
        logger.info(f"Models Agreement Rate: {selection_analysis['similarity']:.2%}")
        logger.info(f"Common Samples: {selection_analysis['common_samples_count']}")
        logger.info("\nGroup Distribution in Common Selections:")
        for group, stats in selection_analysis['group_distribution'].items():
            logger.info(f"Group {group}: {stats['count']} samples ({stats['percentage']:.2%})")
        logger.info("-" * 50)
        
    
    logger.info(f'val_maxAcc: {val_maxAcc}')
    logger.info(f'test_maxAcc: {test_maxAcc}')
    logger.info(f'val_maxAcc: {max(val_maxAcc)}')
    logger.info(f'test_maxAcc: {max(test_maxAcc)}')
    return epoch_statistics

def trainCT(model_args, data_args, training_args, method_args):
    
    start_time = time.time()
    # 配置参数
    config = {
        'train_file': data_args.train_file_path,
        'val_file': data_args.eval_file_path,
        'test_file': data_args.test_file_path,
        'model_names': [method_args.model1_name, method_args.model2_name],
        'batch_size': data_args.batch_size,
        'max_length': data_args.max_sentence_len,
        'num_epochs': training_args.train_epochs,
        'num_labels': model_args.num_classes,
        'device': model_args.device,
        'forget_rate': method_args.forget_rate  # 添加遗忘率参数
    }
    
    # 创建数据加载器
    train_loader, _ = create_dual_dataloader(
        config['train_file'],
        config['model_names'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        shuffle=True
    )
    
    val_loader, _ = create_dual_dataloader(
        config['val_file'],
        config['model_names'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        shuffle=False
    )
    
    test_loader, _ = create_dual_dataloader(
        config['test_file'],
        config['model_names'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        shuffle=False
    )
    logger.info(f'{config["model_names"]}')
    # 开始训练
    logger.info(f"Using device: {config['device']}")
    train_dual_models(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_labels=config['num_labels'],
        model_names=config['model_names'],
        num_epochs=config['num_epochs'],
        device=config['device'],
        forget_rate=config['forget_rate']  # 传入遗忘率参数
    )
    
    end_time = time.time()
    logger.info(f"训练完成，用时{end_time - start_time:.2f}秒")
