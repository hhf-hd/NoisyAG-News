import torch
import logging
from transformers import AutoModelForSequenceClassification
from collections import defaultdict
import os 
from torch.optim import AdamW, SGD, Adam, RMSprop, Adadelta, Adagrad
from typing import Dict, Any, Optional
import numpy as np
import json
import h5py
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score  

logger = logging.getLogger(__name__)


def loss_gls(logits, labels, smooth_rate=0.1):
    """
    Generalized Label Smoothing (GLS) loss function, including both LS and NLS.
    
    Parameters:
    - logits: Model outputs (before softmax), shape [batch_size, num_classes]
    - labels: True class labels (tensor of shape [batch_size])
    - smooth_rate: Label smoothing rate (can be positive for LS or negative for NLS)
    
    Returns:
    - GLS loss value (scalar)
    """
    confidence = 1.0 - smooth_rate  # Target class confidence
    logprobs = F.log_softmax(logits, dim=-1)  # Log-probabilities
    
    # Negative log-likelihood loss for target class
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
    
    # Uniform smoothing loss (negative mean log probability)
    smooth_loss = -logprobs.mean(dim=-1)
    
    # GLS loss computation
    loss = confidence * nll_loss + smooth_rate * smooth_loss
    
    return loss.mean()  # Compute batch mean loss


class NegativeLabelSmoothing(nn.Module):
    """
    负标签平滑损失函数
    与普通的标签平滑不同，负标签平滑只对负类标签施加平滑
    正类标签保持不变
    """
    def __init__(self, smoothing=0.1):
        super(NegativeLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, logits, target):
        n_classes = logits.size(-1)
        
        # 创建one-hot编码
        with torch.no_grad():
            target_one_hot = torch.zeros_like(logits)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            
        # 负标签平滑：正类标签不变，负类标签变为 smoothing/(n_classes-1)
        smoothed_target = target_one_hot * 1.0 - \
                         (1 - target_one_hot) * (self.smoothing / (n_classes - 1))
                         
        loss = torch.sum(-smoothed_target * F.log_softmax(logits, dim=-1), dim=-1)
        return loss.mean()

def get_optimizer(optimizer_name: str, 
                 model_params,
                 training_args,
                 optimizer_config: Optional[Dict[str, Any]] = None):
    """
    获取优化器实例
    Args:
        optimizer_name: 优化器名称
        model_params: 模型参数
        training_args: 训练参数
        optimizer_config: 优化器配置字典，包含学习率等参数
    Returns:
        torch.optim.Optimizer: 优化器实例
    """
    if optimizer_config is None:
        optimizer_config = {}
    
    # 设置默认参数
    default_config = {
        'lr': training_args.learning_rate,
        'weight_decay': 0.0,
    }
    
    # 合并默认参数和用户配置
    config = {**default_config, **optimizer_config}
    
    # 优化器特定的默认参数
    optimizer_specific_defaults = {
        'adam': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': False
        },
        'adamw': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': False
        },
        'sgd': {
            'momentum': 0.0,
            'dampening': 0,
            'nesterov': False
        },
        'rmsprop': {
            'alpha': 0.99,
            'eps': 1e-8,
            'momentum': 0,
            'centered': False
        },
        'adagrad': {
            'lr_decay': 0,
            'eps': 1e-10,
        },
        'adadelta': {
            'rho': 0.9,
            'eps': 1e-6,
        }
    }
    
    # 获取优化器特定的默认参数
    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_specific_defaults:
        specific_defaults = optimizer_specific_defaults[optimizer_name]
        # 合并特定优化器的默认参数和用户配置
        config = {**specific_defaults, **config}
    
    # 优化器映射
    optimizers = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Supported optimizers are: {list(optimizers.keys())}"
        )
    
    # 创建优化器实例
    try:
        optimizer = optimizers[optimizer_name](model_params, **config)
    except TypeError as e:
        raise ValueError(f"Invalid parameters for {optimizer_name}: {str(e)}")
    
    return optimizer

def train_NLS(train_dataloader, eval_dataloader, test_dataloader, model_args, data_args, training_args, method_args):
    """
    使用广义标签平滑(GLS)的训练函数
    """
    start_time = time.time()
    # 初始化模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.pretrained_model_name_or_path, 
        num_labels=model_args.num_classes
    )
    device = model_args.device
    model.to(device)
    logger.info(f"模型初始化完成")
    
    # 创建保存文件夹
    folderName = str(data_args.dataType) + "-NLS-" + str(method_args.negative_smoothing) + "_" + "BS" + str(data_args.batch_size) + "_" + "SEED" + str(training_args.seed) + "_" + "OPT" + str(training_args.optimizer)
    logger.info(f"保存在{folderName}文件夹中")
    pwd = os.getcwd()
    folderPath = os.path.join(pwd, "NLS", folderName)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    # 定义优化器
    optimizer = get_optimizer(training_args.optimizer, model.parameters(), training_args)
    logger.info(f"优化器初始化完成")
    
    # 初始化数据存储字典
    trainData2save = {}
    evalData2save = {}
    testData2save = {}
    
    # 添加进度监控
    total_batches = len(train_dataloader)
    progress_interval = max(1, total_batches // 20)  # 5%的间隔
    
    # 添加时间监控
    epoch_start_time = time.time()
    
    for epoch in range(training_args.train_epochs):
        batch_start_time = time.time()
        logger.info(f"\n开始训练第{epoch+1}/{training_args.train_epochs}轮")
        
        # 初始化当前epoch的数据收集器
        epoch_samples = {
            'sample_ids': [],
            'clean_labels': [],
            'noisy_labels': [],
            'groups': [],
            'texts': [],
            'features': [],
            'softmax_outputs': []
        }
        
        train_loss = 0.0
        model.train()
        
        for index, batch in enumerate(train_dataloader):
            # 计算当前进度百分比
            progress = (index + 1) / total_batches * 100
            
            # 每5%显示一次进度条
            if (index + 1) % progress_interval == 0:
                # 计算预估剩余时间
                elapsed_time = time.time() - batch_start_time
                batches_remaining = total_batches - (index + 1)
                estimated_time = (elapsed_time / (index + 1)) * batches_remaining
                
                progress_bar = create_progress_bar(progress)
                logger.info(f"训练进度: {progress_bar} {progress:.1f}% "
                          f"[{index + 1}/{total_batches}] "
                          f"loss={train_loss/(index + 1):.4f} "
                          f"预计剩余时间: {format_time(estimated_time)}")
            
            # 数据移动到设备
            inputs = batch['input_ids'].to(device)
            labels = batch['noisy_label'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gtLabel = batch['clean_label'].to(device)
            group = batch['group'].to(device)
            text_list = batch['text']
            sample_id = batch['sample_id'].to(device)

            # 前向传播
            outputs = model(
                inputs, 
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # 使用loss_gls计算损失
            logits = outputs.logits
            loss = loss_gls(logits, labels, smooth_rate=method_args.negative_smoothing)
            
            # 计算softmax输出
            softmax_outputs = torch.softmax(logits, dim=1)
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 提取特征
            last_hidden_state = outputs.hidden_states[-1]
            cls_representation = last_hidden_state[:, 0, :].detach()

            # 收集训练数据
            epoch_samples['sample_ids'].append(sample_id.cpu().numpy())
            epoch_samples['clean_labels'].append(gtLabel.cpu().numpy())
            epoch_samples['noisy_labels'].append(labels.cpu().numpy())
            epoch_samples['groups'].append(group.cpu().numpy())
            epoch_samples['texts'].extend(text_list)
            epoch_samples['features'].append(cls_representation.cpu().numpy())
            epoch_samples['softmax_outputs'].append(softmax_outputs.detach().cpu().numpy())

            # 修改日志输出频率
            if index % 100 == 0:
                logger.info(f"详细信息 - batch={index}/{total_batches} loss={loss.item():.4f}")

            # 每10个batch进行验证和测试
            if index % 200 == 0 and index != 0:
                logger.info(f"训练进度：epoch={epoch} batch={index} loss={loss.item():.4f}")
                
                # 验证集评估
                eval_samples = {
                    'sample_ids': [], 'clean_labels': [], 'noisy_labels': [],
                    'groups': [], 'texts': [], 'features': [], 'softmax_outputs': []
                }
                
                model.eval()
                evalGTLabel = []
                evalPredLabel = []
                evalNoisyLabel = []
                with torch.no_grad():
                    for eval_batch in eval_dataloader:
                        # 处理验证集数据
                        eval_inputs = eval_batch['input_ids'].to(device)
                        eval_labels = eval_batch['noisy_label'].to(device)
                        eval_attention = eval_batch['attention_mask'].to(device)
                        eval_gtLabel = eval_batch['clean_label'].to(device)
                        eval_group = eval_batch['group'].to(device)
                        eval_text_list = eval_batch['text']
                        eval_sample_id = eval_batch['sample_id'].to(device)
                        
                        # 前向传播
                        eval_outputs = model(eval_inputs, 
                                          attention_mask=eval_attention,
                                          output_hidden_states=True)
                        
                        # 计算softmax
                        eval_softmax = torch.softmax(eval_outputs.logits, dim=1)
                        
                        # 提取特征
                        eval_features = eval_outputs.hidden_states[-1][:, 0, :]
                        
                        # 收集验证集数据
                        eval_samples['sample_ids'].append(eval_sample_id.cpu().numpy())
                        eval_samples['clean_labels'].append(eval_gtLabel.cpu().numpy())
                        eval_samples['noisy_labels'].append(eval_labels.cpu().numpy())
                        eval_samples['groups'].append(eval_group.cpu().numpy())
                        eval_samples['texts'].extend(eval_text_list)
                        eval_samples['features'].append(eval_features.cpu().numpy())
                        eval_samples['softmax_outputs'].append(eval_softmax.cpu().numpy())
                        evalGTLabel.extend(eval_gtLabel.cpu().numpy())
                        evalPredLabel.extend(eval_softmax.argmax(dim=1).cpu().numpy())
                        evalNoisyLabel.extend(eval_labels.cpu().numpy())
                    
                    gtAcc = accuracy_score(evalGTLabel, evalPredLabel)
                    noisyAcc = accuracy_score(evalNoisyLabel, evalPredLabel)
                    logger.info(f"验证集准确率: gtAcc={gtAcc:.4f} noisyAcc={noisyAcc:.4f}")
                    # 测试集评估
                    test_samples = {
                        'sample_ids': [], 'clean_labels': [], 'noisy_labels': [],
                        'groups': [], 'texts': [], 'features': [], 'softmax_outputs': []
                    }
                    
                    testGTLabel = []
                    testPredLabel = []
                    testNoisyLabel = []
                    
                    for test_batch in test_dataloader:
                        # 处理测试集数据
                        test_inputs = test_batch['input_ids'].to(device)
                        test_labels = test_batch['noisy_label'].to(device)
                        test_attention = test_batch['attention_mask'].to(device)
                        test_gtLabel = test_batch['clean_label'].to(device)
                        test_group = test_batch['group'].to(device)
                        test_text_list = test_batch['text']
                        test_sample_id = test_batch['sample_id'].to(device)
                        
                        # 前向传播
                        test_outputs = model(test_inputs, 
                                          attention_mask=test_attention,
                                          output_hidden_states=True)
                        
                        # 计算softmax
                        test_softmax = torch.softmax(test_outputs.logits, dim=1)
                        
                        # 提取特征
                        test_features = test_outputs.hidden_states[-1][:, 0, :]
                        
                        # 收集测试集数据
                        test_samples['sample_ids'].append(test_sample_id.cpu().numpy())
                        test_samples['clean_labels'].append(test_gtLabel.cpu().numpy())
                        test_samples['noisy_labels'].append(test_labels.cpu().numpy())
                        test_samples['groups'].append(test_group.cpu().numpy())
                        test_samples['texts'].extend(test_text_list)
                        test_samples['features'].append(test_features.cpu().numpy())
                        test_samples['softmax_outputs'].append(test_softmax.cpu().numpy())
                        testGTLabel.extend(test_gtLabel.cpu().numpy())
                        testPredLabel.extend(test_softmax.argmax(dim=1).cpu().numpy())
                        testNoisyLabel.extend(test_labels.cpu().numpy())
                    gtTestAcc = accuracy_score(testGTLabel, testPredLabel)
                    noisyTestAcc = accuracy_score(testNoisyLabel, testPredLabel)
                    logger.info(f"测试集准确率: gtAcc={gtTestAcc:.4f} noisyAcc={noisyTestAcc:.4f}")
                    
                # 保存验证集和测试集数据
                step_key = f"epoch{epoch}_batch{index}"
                
                # 合并验证集数据
                evalData2save[step_key] = {
                    'sample_ids': np.concatenate(eval_samples['sample_ids'], axis=0),
                    'clean_labels': np.concatenate(eval_samples['clean_labels'], axis=0),
                    'noisy_labels': np.concatenate(eval_samples['noisy_labels'], axis=0),
                    'groups': np.concatenate(eval_samples['groups'], axis=0),
                    'texts': eval_samples['texts'],
                    'features': np.concatenate(eval_samples['features'], axis=0),
                    'softmax_outputs': np.concatenate(eval_samples['softmax_outputs'], axis=0)
                }
                
                # 合并测试集数据
                testData2save[step_key] = {
                    'sample_ids': np.concatenate(test_samples['sample_ids'], axis=0),
                    'clean_labels': np.concatenate(test_samples['clean_labels'], axis=0),
                    'noisy_labels': np.concatenate(test_samples['noisy_labels'], axis=0),
                    'groups': np.concatenate(test_samples['groups'], axis=0),
                    'texts': test_samples['texts'],
                    'features': np.concatenate(test_samples['features'], axis=0),
                    'softmax_outputs': np.concatenate(test_samples['softmax_outputs'], axis=0)
                }
                
                model.train()

        # 合并当前epoch的训练数据
        trainData2save[epoch] = {
            'sample_ids': np.concatenate(epoch_samples['sample_ids'], axis=0),
            'clean_labels': np.concatenate(epoch_samples['clean_labels'], axis=0),
            'noisy_labels': np.concatenate(epoch_samples['noisy_labels'], axis=0),
            'groups': np.concatenate(epoch_samples['groups'], axis=0),
            'texts': epoch_samples['texts'],
            'features': np.concatenate(epoch_samples['features'], axis=0),
            'softmax_outputs': np.concatenate(epoch_samples['softmax_outputs'], axis=0)
        }

        # Epoch结束时的统计
        avg_loss = train_loss/len(train_dataloader)
        logger.info(f"\nEpoch {epoch+1} 完成 - 平均损失={avg_loss:.4f}")
    
    # 保存所有数据
    save_path = os.path.join(folderPath, "train_data.h5")
    with h5py.File(save_path, 'w') as f:
        for epoch, data in trainData2save.items():
            group = f.create_group(f'epoch_{epoch}')
            for key, value in data.items():
                if key != 'texts':
                    group.create_dataset(key, data=value, compression='gzip')
                else:
                    # 将文本列表转换为字节字符串数组
                    text_data = np.array(value, dtype='S')
                    group.create_dataset(key, data=text_data, compression='gzip')
    
    # 保存验证集数据
    save_path = os.path.join(folderPath, "eval_data.h5")
    with h5py.File(save_path, 'w') as f:
        for step, data in evalData2save.items():
            group = f.create_group(step)
            for key, value in data.items():
                if key != 'texts':
                    group.create_dataset(key, data=value, compression='gzip')
                else:
                    text_data = np.array(value, dtype='S')
                    group.create_dataset(key, data=text_data, compression='gzip')
    
    # 保存测试集数据
    save_path = os.path.join(folderPath, "test_data.h5")
    with h5py.File(save_path, 'w') as f:
        for step, data in testData2save.items():
            group = f.create_group(step)
            for key, value in data.items():
                if key != 'texts':
                    group.create_dataset(key, data=value, compression='gzip')
                else:
                    text_data = np.array(value, dtype='S')
                    group.create_dataset(key, data=text_data, compression='gzip')
    
    end_time = time.time()
    logger.info(f"训练完成，用时{end_time - start_time:.2f}秒")
  
# 添加这两个辅助函数
def create_progress_bar(percentage, width=50):
    """
    创建ASCII进度条
    Args:
        percentage: 完成百分比 (0-100)
        width: 进度条宽度
    Returns:
        str: ASCII进度条
    """
    filled = int(width * percentage / 100)
    bar = '█' * filled + '-' * (width - filled)
    return f"[{bar}]"

def format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
  