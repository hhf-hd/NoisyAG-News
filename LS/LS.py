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
logger = logging.getLogger(__name__)

class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失函数
    """
    def __init__(self, smoothing):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_optimizer(optimizer_name: str, 
                 model_params,training_args,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 ):
    """
    获取优化器实例
    Args:
        optimizer_name: 优化器名称
        model_params: 模型参数
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

def train_LS(train_dataloader, eval_dataloader, test_dataloader, model_args, data_args, training_args, method_args):
    """
    使用Label Smoothing的训练函数
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
    folderName = str(data_args.dataType) + "_LS" + str(method_args.label_smoothing) + "_" + "BS" + str(data_args.batch_size) + "_" + "SEED" + str(training_args.seed) + "_" + "OPT" + str(training_args.optimizer)
    pwd = os.getcwd()
    folderPath = os.path.join(pwd, "LS", folderName)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    # 定义优化器和损失函数
    optimizer = get_optimizer(training_args.optimizer, model.parameters(), training_args)
    criterion = LabelSmoothingCrossEntropy(smoothing=method_args.label_smoothing)
    logger.info(f"优化器和损失函数初始化完成")
    
    # 初始化数据存储字典
    trainData2save = {}  # 每个epoch结束时的训练数据
    evalData2save = {}   # 动态记录的验证数据
    testData2save = {}   # 动态记录的测试数据
    trainDynamicData2save = {}  # 动态记录的训练数据
    # 添加进度监控
    total_batches = len(train_dataloader)
    progress_interval = max(1, total_batches // 20)  # 5%的间隔
    
    
    # 动态记录策略参数
    base_interval = 50  # 初始记录间隔
    max_interval = 500  # 最大记录间隔
    interval_growth_rate = 1.5  # 间隔增长率
    current_interval = base_interval  # 当前记录间隔
    last_record_batch = 0  # 上次记录的batch编号
    
    batchCount = 0
    
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
        
        epoch_start_time = time.time()
        
        for index, batch in enumerate(train_dataloader):
            
            batchCount += 1
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
            
            # 计算损失
            logits = outputs.logits
            loss = criterion(logits, labels)
            
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
            
            should_record = False
            
            if batchCount < 600 and batchCount % 100 == 0:
                should_record = True
                
            if batchCount > 600 and batchCount < 2000 and batchCount % 200 == 0:
                should_record = True
            
            if batchCount > 2000  and batchCount % 500 == 0:
                should_record = True
            
            
                
            if should_record:
                logger.info(f"执行动态记录 - epoch={epoch} batch={batchCount} ")
                record_start_time = time.time()
                
                # 更新记录间隔
                current_interval = min(int(current_interval * interval_growth_rate), max_interval)
                # last_record_batch = index
                
                model.eval()
                with torch.no_grad():
                    if data_args.saveTrainInfo:
                        # 对训练集进行动态记录
                        train_dynamic_samples = {
                            'sample_ids': [], 'clean_labels': [], 'noisy_labels': [],
                            'groups': [], 'texts': [], 'features': [], 'softmax_outputs': []
                        }
                        
                        for train_eval_batch in train_dataloader:
                            train_inputs = train_eval_batch['input_ids'].to(device)
                            train_labels = train_eval_batch['noisy_label'].to(device)
                            train_attention = train_eval_batch['attention_mask'].to(device)
                            train_gtLabel = train_eval_batch['clean_label'].to(device)
                            train_group = train_eval_batch['group'].to(device)
                            train_text_list = train_eval_batch['text']
                            train_sample_id = train_eval_batch['sample_id'].to(device)
                            
                            train_outputs = model(train_inputs, 
                                            attention_mask=train_attention,
                                            output_hidden_states=True)
                            
                            train_softmax = torch.softmax(train_outputs.logits, dim=1)
                            train_features = train_outputs.hidden_states[-1][:, 0, :]
                            
                            train_dynamic_samples['sample_ids'].append(train_sample_id.cpu().numpy())
                            train_dynamic_samples['clean_labels'].append(train_gtLabel.cpu().numpy())
                            train_dynamic_samples['noisy_labels'].append(train_labels.cpu().numpy())
                            train_dynamic_samples['groups'].append(train_group.cpu().numpy())
                            train_dynamic_samples['texts'].extend(train_text_list)
                            train_dynamic_samples['features'].append(train_features.cpu().numpy())
                            train_dynamic_samples['softmax_outputs'].append(train_softmax.cpu().numpy())
                    
                    # 对验证集进行记录
                    eval_samples = {
                        'sample_ids': [], 'clean_labels': [], 'noisy_labels': [],
                        'groups': [], 'texts': [], 'features': [], 'softmax_outputs': []
                    }
                    
                    for eval_batch in eval_dataloader:
                        eval_inputs = eval_batch['input_ids'].to(device)
                        eval_labels = eval_batch['noisy_label'].to(device)
                        eval_attention = eval_batch['attention_mask'].to(device)
                        eval_gtLabel = eval_batch['clean_label'].to(device)
                        eval_group = eval_batch['group'].to(device)
                        eval_text_list = eval_batch['text']
                        eval_sample_id = eval_batch['sample_id'].to(device)
                        
                        eval_outputs = model(eval_inputs, 
                                          attention_mask=eval_attention,
                                          output_hidden_states=True)
                        
                        eval_softmax = torch.softmax(eval_outputs.logits, dim=1)
                        eval_features = eval_outputs.hidden_states[-1][:, 0, :]
                        
                        eval_samples['sample_ids'].append(eval_sample_id.cpu().numpy())
                        eval_samples['clean_labels'].append(eval_gtLabel.cpu().numpy())
                        eval_samples['noisy_labels'].append(eval_labels.cpu().numpy())
                        eval_samples['groups'].append(eval_group.cpu().numpy())
                        eval_samples['texts'].extend(eval_text_list)
                        eval_samples['features'].append(eval_features.cpu().numpy())
                        eval_samples['softmax_outputs'].append(eval_softmax.cpu().numpy())
                    
                    # 对测试集进行记录
                    test_samples = {
                        'sample_ids': [], 'clean_labels': [], 'noisy_labels': [],
                        'groups': [], 'texts': [], 'features': [], 'softmax_outputs': []
                    }
                    
                    for test_batch in test_dataloader:
                        test_inputs = test_batch['input_ids'].to(device)
                        test_labels = test_batch['noisy_label'].to(device)
                        test_attention = test_batch['attention_mask'].to(device)
                        test_gtLabel = test_batch['clean_label'].to(device)
                        test_group = test_batch['group'].to(device)
                        test_text_list = test_batch['text']
                        test_sample_id = test_batch['sample_id'].to(device)
                        
                        test_outputs = model(test_inputs, 
                                          attention_mask=test_attention,
                                          output_hidden_states=True)
                        
                        test_softmax = torch.softmax(test_outputs.logits, dim=1)
                        test_features = test_outputs.hidden_states[-1][:, 0, :]
                        
                        test_samples['sample_ids'].append(test_sample_id.cpu().numpy())
                        test_samples['clean_labels'].append(test_gtLabel.cpu().numpy())
                        test_samples['noisy_labels'].append(test_labels.cpu().numpy())
                        test_samples['groups'].append(test_group.cpu().numpy())
                        test_samples['texts'].extend(test_text_list)
                        test_samples['features'].append(test_features.cpu().numpy())
                        test_samples['softmax_outputs'].append(test_softmax.cpu().numpy())
                
                # 保存动态记录数据
                step_key = f"epoch{epoch}_batch{batchCount}"
                
                if data_args.saveTrainInfo:
                    # 保存训练集动态记录
                    trainDynamicData2save[step_key] = {
                        'sample_ids': np.concatenate(train_dynamic_samples['sample_ids'], axis=0),
                        'clean_labels': np.concatenate(train_dynamic_samples['clean_labels'], axis=0),
                        'noisy_labels': np.concatenate(train_dynamic_samples['noisy_labels'], axis=0),
                        'groups': np.concatenate(train_dynamic_samples['groups'], axis=0),
                        'texts': train_dynamic_samples['texts'],
                        'features': np.concatenate(train_dynamic_samples['features'], axis=0),
                        'softmax_outputs': np.concatenate(train_dynamic_samples['softmax_outputs'], axis=0)
                    }
                
                # 保存验证集记录
                evalData2save[step_key] = {
                    'sample_ids': np.concatenate(eval_samples['sample_ids'], axis=0),
                    'clean_labels': np.concatenate(eval_samples['clean_labels'], axis=0),
                    'noisy_labels': np.concatenate(eval_samples['noisy_labels'], axis=0),
                    'groups': np.concatenate(eval_samples['groups'], axis=0),
                    'texts': eval_samples['texts'],
                    'features': np.concatenate(eval_samples['features'], axis=0),
                    'softmax_outputs': np.concatenate(eval_samples['softmax_outputs'], axis=0)
                }
                
                # 保存测试集记录
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
                
                record_end_time = time.time()
                record_duration = record_end_time - record_start_time
                logger.info(f"执行动态记录 - epoch={epoch} batch={batchCount} 用时{record_duration:.2f}秒")
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch+1} 完成 - 用时{epoch_duration:.2f}秒")
        
        # 保存每个epoch结束时的训练数据
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
    
    # 训练结束后，一次性保存所有数据
    logger.info("开始保存所有数据...")
    
    # 1. 保存训练数据（每个epoch结束时的状态）
    save_path = os.path.join(folderPath, "train.h5")
    with h5py.File(save_path, 'w') as f:
        for epoch, data in trainData2save.items():
            group = f.create_group(f'epoch_{epoch}')
            for key, value in data.items():
                if key != 'texts':
                    group.create_dataset(key, data=value, compression='gzip')
                else:
                    text_data = np.array(value, dtype='S')
                    group.create_dataset(key, data=text_data, compression='gzip')
    
    # 2. 保存验证集数据（动态记录）
    save_path = os.path.join(folderPath, "eval.h5")
    with h5py.File(save_path, 'w') as f:
        for step, data in evalData2save.items():
            group = f.create_group(step)
            for key, value in data.items():
                if key != 'texts':
                    group.create_dataset(key, data=value, compression='gzip')
                else:
                    text_data = np.array(value, dtype='S')
                    group.create_dataset(key, data=text_data, compression='gzip')
    
    # 3. 保存测试集数据（动态记录）
    save_path = os.path.join(folderPath, "test.h5")
    with h5py.File(save_path, 'w') as f:
        for step, data in testData2save.items():
            group = f.create_group(step)
            for key, value in data.items():
                if key != 'texts':
                    group.create_dataset(key, data=value, compression='gzip')
                else:
                    text_data = np.array(value, dtype='S')
                    group.create_dataset(key, data=text_data, compression='gzip')
    if data_args.saveTrainInfo:
        # 4. 保存训练集动态记录数据
        save_path = os.path.join(folderPath, "train_dynamic.h5")
        with h5py.File(save_path, 'w') as f:
            for step, data in trainDynamicData2save.items():
                group = f.create_group(step)
                for key, value in data.items():
                    if key != 'texts':
                        group.create_dataset(key, data=value, compression='gzip')
                    else:
                        text_data = np.array(value, dtype='S')
                        group.create_dataset(key, data=text_data, compression='gzip')
    
    end_time = time.time()
    logger.info(f"训练完成，总用时{end_time - start_time:.2f}秒")
    logger.info("所有数据已保存到以下文件：")
    logger.info(f"1. 训练数据：{os.path.join(folderPath, 'train.h5')}")
    logger.info(f"2. 验证数据：{os.path.join(folderPath, 'eval.h5')}")
    logger.info(f"3. 测试数据：{os.path.join(folderPath, 'test.h5')}")
    logger.info(f"4. 训练动态记录：{os.path.join(folderPath, 'train_dynamic.h5')}")

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
  