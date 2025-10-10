from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os, logging, warnings
import torch
import torch.nn as nn

from torch.optim import AdamW, SGD, Adam, RMSprop, Adadelta, Adagrad

from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch.nn.functional as F
import h5py
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')


def metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')  # 每一类预测对的占比取平均
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def exp_decay(current, start=1.0, end=0.0, exp=3, rampup_length=100):  # 从1指数衰减到0
    current = np.clip(current / rampup_length, 0.0, 1.0)  # 限制到0-1之间
    
    return end + (start - end) * np.exp(-current * exp)

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
        'lr': training_args.learning_rate, # 学习率
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
        logger.info(f"优化器初始化完成: {optimizer_name}")
    except TypeError as e:
        raise ValueError(f"Invalid parameters for {optimizer_name}: {str(e)}")
    
    return optimizer


class expDecayTrainer:
    def __init__(self, model, train_data=None, eval_data=None, test_data=None, model_args=None,data_args=None, training_args=None,method_args=None):
        self.model = model.cuda()
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.model_args = model_args
        self.data_args = data_args  
        self.training_args = training_args
        self.method_args = method_args

       
        self.optimizer = get_optimizer(training_args.optimizer, model.parameters(),training_args)

        # 添加数据保存路径
        self.folderName = f"{data_args.dataType}_exp{method_args.exp}_ph{method_args.p_threshold}_temp{method_args.temp}_lambda{method_args.lambda_r}"
        pwd = os.getcwd()
        self.folderPath = os.path.join(pwd, "expDecay", self.folderName)
        if not os.path.exists(self.folderPath):
            os.makedirs(self.folderPath)


    def dynamic_train(self):
        test_best_l = []
        test_last_l = []

        test_best = 0.0

        train_loader = self.train_data.run("all")
        eval_loader = self.eval_data.run("all")  # 之前没有

        train_iters = self.training_args.train_epochs * len(train_loader)

        train_iter = iter(train_loader)

        
        iter_len = len(train_loader)
        eval_steps = []
        for tt in range(self.training_args.split_num):
            eval_steps.append(int(tt * iter_len / self.training_args.split_num))
        logger.info(f"eval_steps {eval_steps}")
    
        logger.info(f"iter_len {iter_len}")
        
        logger.info(f"train_iters {train_iters}")
        
        logger.info("Training begin...")
        w_x_records = []

        # 初始化数据收集字典
        trainData2save = {}  # 每个batch的训练数据
        evalData2save = {}   # 每100个batch的验证数据
        testData2save = {}   # 每100个batch的测试数据
        
        train_gt_acc_records = []
        train_noisy_acc_records = []
        eval_gt_acc_records = []
        eval_noisy_acc_records = []
        test_gt_acc_records = []
        test_noisy_acc_records = []
        
        for i in range(0, train_iters):
            logger.info(f"训练进度：iter={i}/{train_iters}")

            self.model.train()

            try:
                # data = train_iter.next()
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                # data = train_iter.next()
                data = next(train_iter)

            input_ids, att_mask, labels_id, true_labels, groups, index = [Variable(elem.cuda()) for elem in data]
            labels = F.one_hot(labels_id, num_classes=self.model_args.num_classes)

            self.model.eval()
            with torch.no_grad():
                # Predict labels for all data.
                out_x = self.model(input_ids, att_mask)
                p_x = torch.softmax(out_x, dim=1)
                # p_threshold用来控制下限 exp控制函数衰减幅度
                w_x = exp_decay(i, start=1, end=self.method_args.p_threshold, exp=self.method_args.exp,
                                rampup_length=self.training_args.train_epochs * len(
                                    train_loader))  # 在整个训练期间衰减逐渐衰减
                # W_x表示了真实标签的比例，最后会接近p_threshold
                w_x_records.append(w_x)
                p_x = (1 - w_x) * p_x + w_x * labels

                pt_x = p_x ** (1 / self.method_args.temp)
                targets_x = pt_x / pt_x.sum(dim=1, keepdim=True)
                targets_x = targets_x.detach()

            self.model.train()

            # norm train
            sents_x = self.model.get_sentence_embedding(input_ids, att_mask)
            sents_x2 = self.model.get_sentence_embedding(input_ids, att_mask)
            logits_x = self.model.classify(sents_x)
            logits_x2 = self.model.classify(sents_x2)
            loss_cl = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=-1) * targets_x, dim=-1))
            kl_loss = compute_kl_loss(logits_x, logits_x2)

            # 避免崩溃
            prior = torch.ones(self.model_args.num_classes) / self.model_args.num_classes
            prior = prior.cuda()
            pred_mean = torch.softmax(torch.cat([logits_x, logits_x2], dim=0), dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))
            cl_weight = 1.0  # 或其他合适的值
            loss = cl_weight * loss_cl + kl_loss * self.method_args.lambda_r + penalty
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()


            # 收集每个batch的训练数据时也应该记录更多信息
            batch_samples = {
                'sample_ids': index.cpu().numpy(),
                'clean_labels': true_labels.cpu().numpy(),
                'noisy_labels': labels_id.cpu().numpy(),
                'groups': groups.cpu().numpy(),
                'features': sents_x.detach().cpu().numpy(),
                'softmax_outputs': torch.softmax(logits_x, dim=1).detach().cpu().numpy(),
                'w_x': w_x,  # 记录当前batch的权重
                'loss': loss.item(),  # 记录损失值
                'kl_loss': kl_loss.item(),  # 记录KL损失
                'penalty': penalty.item(),  # 记录惩罚项
            }
            trainData2save[f"batch_{i}"] = batch_samples
            evaluateFlag = False

            # 每100个batch进行验证和测试数据收集
            if i != 0 and i % 100 == 0 and i < 1000:
                evaluateFlag = True
            if i != 0 and i % 200 == 0 and i >= 1000 and i < 3000:
                evaluateFlag = True
            if i != 0 and i % 300 == 0 and i >= 3000 :
                evaluateFlag = True
           
                
            if evaluateFlag:
                
                evaluateFlag = False
                logger.info(f"@@@@@@训练进度：iter={i}/{train_iters}")
                
                # 对训练数据的评估也使用eval_train
                clean_right, clean_wrong, noisy_right, noisy_noise, noisy_other, train_acc, group_stats = self.eval_train(train_loader)
                train_samples = {
                    'clean_right': clean_right,
                    'clean_wrong': clean_wrong,
                    'noisy_right': noisy_right,
                    'noisy_noise': noisy_noise,
                    'noisy_other': noisy_other,
                    'accuracy': train_acc,
                    'group_stats': group_stats
                }
                train_gt_acc_records.append(clean_right)
                train_noisy_acc_records.append(noisy_right)
                trainData2save[f"eval_at_iter_{i}"] = train_samples
                
                # 收集验证数据
                eval_samples = self.collect_data(eval_loader,data_type='eval')
                eval_gt_acc_records.append(eval_samples['accuracy'])
                eval_noisy_acc_records.append(eval_samples['noisy_accuracy'])
                evalData2save[f"iter_{i}"] = eval_samples
                
                # 收集测试数据
                test_samples = self.collect_data(self.test_data.run("all"),data_type='test')
                test_gt_acc_records.append(test_samples['accuracy'])
                test_noisy_acc_records.append(test_samples['noisy_accuracy'])
                testData2save[f"iter_{i}"] = test_samples
                
                logger.info(f"当前训练准确率（干净样本）: {clean_right:.4f}")
                logger.info(f"当前训练准确率（噪声样本正确预测）: {noisy_right:.4f}")
                # logger.info(f"当前验证准确率: {eval_samples['accuracy']:.4f}")
                # logger.info(f"当前测试准确率: {test_samples['accuracy']:.4f}")
        
        # 保存收集的数据
        # self.save_collected_data(trainData2save, "train_data.h5")
        # self.save_collected_data(evalData2save, "eval_data.h5")
        # self.save_collected_data(testData2save, "test_data.h5")
        
        logger.info(f"train_gt_acc_records: {train_gt_acc_records}")
        logger.info(f"train_noisy_acc_records: {train_noisy_acc_records}")
        logger.info(f"eval_gt_acc_records: {eval_gt_acc_records}")
        logger.info(f"eval_noisy_acc_records: {eval_noisy_acc_records}")
        logger.info(f"test_gt_acc_records: {test_gt_acc_records}")
        logger.info(f"test_noisy_acc_records: {test_noisy_acc_records}")
        logger.info(f"max train_gt_acc_records: {max(train_gt_acc_records)}")
        logger.info(f"max train_noisy_acc_records: {max(train_noisy_acc_records)}")
        logger.info(f"max eval_gt_acc_records: {max(eval_gt_acc_records)}")
        logger.info(f"max eval_noisy_acc_records: {max(eval_noisy_acc_records)}")
        logger.info(f"max test_gt_acc_records: {max(test_gt_acc_records)}")
        logger.info(f"max test_noisy_acc_records: {max(test_noisy_acc_records)}")
        
        return test_best_l, test_last_l

    def collect_data(self, data_loader,data_type='eval'):
        logger.info(f"#"*36)
        logger.info(f"收集{data_type}数据...")
        """收集数据的辅助函数"""
        self.model.eval()
        samples = {
            'sample_ids': [],
            'clean_labels': [],
            'noisy_labels': [],
            'groups': [],
            'features': [],
            'softmax_outputs': [],
            'accuracy': 0.0
        }
        
        total_correct = 0
        total_noisy_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data in data_loader:
                input_ids, att_mask, labels, true_labels, groups, index = [Variable(elem.cuda()) for elem in data]
                
                # 获取特征和预测
                sents = self.model.get_sentence_embedding(input_ids, att_mask)
                outputs = self.model.classify(sents)
                softmax_outputs = torch.softmax(outputs, dim=1)
                
                # 计算准确率
                pred = torch.argmax(outputs, dim=1)
                total_correct += (pred == true_labels).sum().item()
                total_samples += len(true_labels)
                total_noisy_correct += (pred == labels).sum().item()
                
                # 收集数据
                samples['sample_ids'].append(index.cpu().numpy())
                samples['clean_labels'].append(true_labels.cpu().numpy())
                samples['noisy_labels'].append(labels.cpu().numpy())
                samples['groups'].append(groups.cpu().numpy())
                samples['features'].append(sents.cpu().numpy())
                samples['softmax_outputs'].append(softmax_outputs.cpu().numpy())
        
        # 合并数据
        for key in ['sample_ids', 'clean_labels', 'noisy_labels', 'groups', 'features', 'softmax_outputs']:
            if len(samples[key]) > 0:
                samples[key] = np.concatenate(samples[key], axis=0)
        
        samples['accuracy'] = total_correct / total_samples
        samples['noisy_accuracy'] = total_noisy_correct / total_samples
        logger.info(f"当前验证准确率: {samples['accuracy']:.4f}")
        logger.info(f"当前验证准确率(在噪声标签上): {samples['noisy_accuracy']:.4f}")
        logger.info(f"#"*36)
        # logger.info("\n")
        return samples

    def save_collected_data(self, data_dict, filename):
        """保存收集的数据到h5文件"""
        save_path = os.path.join(self.folderPath, filename)
        with h5py.File(save_path, 'w') as f:
            for step, data in data_dict.items():
                group = f.create_group(step)
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value, compression='gzip')
                    elif isinstance(value, float):
                        group.attrs[key] = value

    
    def evaluate(self, eval_loader=None):
        if eval_loader is None:
            eval_loader = self.eval_data.run("all")
        self.model.eval()
        total_correct = 0
        total_samples = 0
        y_true, y_pred = np.zeros(len(eval_loader.dataset), dtype=int), np.zeros(len(eval_loader.dataset), dtype=int)
        for j, data in enumerate(eval_loader):
            # val_input_ids, val_att, val_labels, _, groups, index = [Variable(elem.cuda()) for elem in data]
            val_input_ids, val_att, val_labels, _, groups, index = data
            val_input_ids = val_input_ids.cuda()
            val_att = val_att.cuda()
            val_labels = val_labels.cuda()
            
            with torch.no_grad():
                index = index.long().cpu().detach().numpy()
                pred = self.model(val_input_ids, val_att).argmax(dim=-1).cpu().detach().numpy()
                val_labels = val_labels.cpu().detach().numpy()
            y_true[index] = val_labels
            y_pred[index] = pred
            total_correct += (pred == val_labels).sum().item()
            total_samples += len(val_labels)

        eval_acc = total_correct / total_samples
        
        logger.info(f"Eval Results in evaluate: Accuracy: {eval_acc:.2%}")
        eval_res = metric(y_true, y_pred)
        logger.info("Eval Results: Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}"
                    .format(eval_res['accuracy'], eval_res['precision'], eval_res['recall'], eval_res['f1']))
        return eval_res['accuracy'], eval_res['f1'], eval_res['recall']

    def save_model(self, comm=None):
        suffix = '.pt'
        if comm:
            suffix = '_' + str(comm) + suffix

        path = self.training_args.model_save_path + suffix
        dir = os.path.dirname(path)
        folder = os.path.exists(dir)

        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not folder:
            os.makedirs(dir)  # makedirs 创建文件时如果路径不存在会创建这个路径
          
        self.model.save_model(path)

    def load_model(self, comm=None):
        logger.info('模型重载中...')  
        suffix = '.pt'
        if comm:
            suffix = '_' + str(comm) + suffix
        path = self.training_args.model_save_path + suffix
        model_state_dict = torch.load(path)
        self.model.load_state_dict(model_state_dict)

    def eval_train(self, train_loader):
        logger.info(f"#"*20)
        logger.info("Evaluating training data...")
        """评估训练数据，分别统计干净样本和噪声样本的表现"""
        self.model.eval()
        y_nois = np.zeros(len(train_loader.dataset), dtype=int)  # 噪声标签
        y_true = np.zeros(len(train_loader.dataset), dtype=int)  # 真实标签
        y_pred = np.zeros(len(train_loader.dataset), dtype=int)  # 预测标签
        y_groups = np.zeros(len(train_loader.dataset), dtype=int)  # 组别信息
        
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                input_ids, att_mask, labels, true_labels, groups, index = [Variable(elem.cuda()) for elem in data]
                
                outputs = self.model(input_ids, att_mask)
                pred = torch.argmax(outputs, dim=-1).cpu().detach().numpy()
                index = index.long().cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                true_labels = true_labels.cpu().detach().numpy()
                groups = groups.cpu().detach().numpy()
                
                y_nois[index] = labels  # 噪声标签
                y_true[index] = true_labels  # 真实标签
                y_pred[index] = pred  # 预测标签
                y_groups[index] = groups  # 组别信息
        
        # 区分干净样本和噪声样本
        clean_index = y_nois == y_true  # 干净样本的索引
        noisy_index = ~clean_index  # 噪声样本的索引

        # 计算各类指标
        # 干净样本中预测正确和错误的比例
        clean_right = (y_pred[clean_index] == y_nois[clean_index]).sum() / max(1, len(y_nois[clean_index]))
        clean_wrong = (y_pred[clean_index] != y_nois[clean_index]).sum() / max(1, len(y_nois[clean_index]))
        
        # 噪声样本中的各种情况
        noisy_right = (y_pred[noisy_index] == y_true[noisy_index]).sum() / max(1, len(y_true[noisy_index]))  # 预测为真实标签
        noisy_noise = (y_pred[noisy_index] == y_nois[noisy_index]).sum() / max(1, len(y_nois[noisy_index]))  # 预测为噪声标签
        noisy_other = ((y_pred[noisy_index] != y_nois[noisy_index]) * 
                       (y_pred[noisy_index] != y_true[noisy_index])).sum() / max(1, len(y_nois[noisy_index]))  # 预测为其他标签

        # 按组统计指标
        group_stats = {}
        unique_groups = np.unique(y_groups)
        
        for group in unique_groups:
            group_idx = (y_groups == group)
            group_clean_idx = group_idx & clean_index
            group_noisy_idx = group_idx & noisy_index
            
            # 该组样本数量
            group_total = np.sum(group_idx)
            # 该组干净样本数量及占比
            group_clean_count = np.sum(group_clean_idx)
            group_clean_ratio = group_clean_count / max(1, group_total)
            # 该组噪声样本数量及占比
            group_noisy_count = np.sum(group_noisy_idx)
            group_noisy_ratio = group_noisy_count / max(1, group_total)
            
            # 该组干净样本中预测正确和错误的比例
            group_clean_right = np.sum(y_pred[group_clean_idx] == y_nois[group_clean_idx]) / max(1, group_clean_count)
            group_clean_wrong = np.sum(y_pred[group_clean_idx] != y_nois[group_clean_idx]) / max(1, group_clean_count)
            
            # 该组噪声样本中预测对应正确标签、噪声标签和都不对应的比例
            group_noisy_right = np.sum(y_pred[group_noisy_idx] == y_true[group_noisy_idx]) / max(1, group_noisy_count)
            group_noisy_noise = np.sum(y_pred[group_noisy_idx] == y_nois[group_noisy_idx]) / max(1, group_noisy_count)
            group_noisy_other = np.sum((y_pred[group_noisy_idx] != y_nois[group_noisy_idx]) & 
                                      (y_pred[group_noisy_idx] != y_true[group_noisy_idx])) / max(1, group_noisy_count)
            
            # 该组总体准确率
            group_accuracy = np.sum(y_pred[group_idx] == y_true[group_idx]) / max(1, group_total)
            
            group_stats[int(group)] = {
                'total': int(group_total),
                'clean_ratio': float(group_clean_ratio),
                'noisy_ratio': float(group_noisy_ratio),
                'clean_right': float(group_clean_right),
                'clean_wrong': float(group_clean_wrong),
                'noisy_right': float(group_noisy_right),
                'noisy_noise': float(group_noisy_noise),
                'noisy_other': float(group_noisy_other),
                'accuracy': float(group_accuracy)
            }
        
        # 计算总体准确率（相对于真实标签）
        train_acc = (y_pred == y_true).mean()
        train_acc_noisy = (y_pred== y_nois).mean()
        logger.info(f"当前训练准确率（总体）: {train_acc:.4f}")
        logger.info(f"当前训练准确率（总体在噪声标签上）: {train_acc_noisy:.4f}")
        logger.info(f"当前训练准确率（干净样本）: {clean_right:.4f}")
        logger.info(f"当前训练准确率（噪声样本正确预测）: {noisy_right:.4f}")
        logger.info(f"当前训练准确率（噪声样本错误预测）: {noisy_noise:.4f}")
        logger.info(f"当前训练准确率（噪声样本错误预测到其他类别）: {noisy_other:.4f}")
        logger.info(f"#"*20)
        
        return clean_right, clean_wrong, noisy_right, noisy_noise, noisy_other, train_acc, group_stats
