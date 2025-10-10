from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.mixture import GaussianMixture
import os, logging, warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import time 
import h5py  # 添加h5py导入


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')


def metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
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


class SelfMixTrainer:
    def __init__(self, model, train_data=None, eval_data=None, test_data=None, model_args=None, data_args=None, training_args=None, method_args=None):
        self.model = model.cuda()
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.method_args = method_args
        if self.training_args is not None:
            self.optimizer = Adam(self.model.parameters(), lr=training_args.learning_rate)
            
        # 创建用于跟踪样本的数据结构
        self.sample_tracking = {
            'epoch': [],          # 所属epoch
            'sample_id': [],      # 样本ID
            'is_labeled': [],     # 是否被分到有标签集
            'gmm_prob': [],       # GMM概率值
            'loss': [],           # 损失值
            'pred': [],           # 预测标签
            'true_label': [],     # 真实标签
            'noisy_label': [],    # 噪声标签
            'softmax': [],        # softmax输出
            'features': [],       # 特征表示
            'group': []           # 组标识
        }
        
        # 创建用于记录训练动态的数据结构
        self.training_dynamics = {
            'epoch': [],
            'total_loss': [],
            'mix_loss': [],
            'pse_loss': [],
            'kl_loss': [],
            'labeled_ratio': [],
            'labeled_good_ratio': [],
            'unlabeled_good_ratio': []
        }
        
        # 创建保存路径
        if self.training_args is not None:
            self.save_dir = os.path.join(os.getcwd(), "selfMix", 
                                        f"{data_args.dataType}_p{self.method_args.p_threshold}_temp{self.method_args.temp}_alpha{self.method_args.alpha}")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
    
    def warmup(self):
        logger.info("***** Warmup stage *****")
        
        train_loader = self.train_data.run("all")
        if self.training_args.warmup_strategy == "epoch":
            warmup_samples = self.training_args.warmup_epochs * len(train_loader.dataset)
            warmup_epochs = self.training_args.warmup_epochs
        elif self.training_args.warmup_strategy == "samples":
            warmup_samples = self.training_args.warmup_samples
            warmup_epochs = self.training_args.warmup_samples // len(train_loader.dataset) + \
                            int(self.training_args.warmup_samples % len(train_loader.dataset) > 0)
        else:
            warmup_samples, warmup_epochs = 0, 0
            
        loss_func = nn.CrossEntropyLoss()
        now_samples = 0
        logger.info(f"warmup_samples: {warmup_samples}")
        logger.info(f"warmup_epochs: {warmup_epochs}")
        for epoch_id in range(1, warmup_epochs + 1):
            logger.info("***** Warmup epoch %d *****", epoch_id)
            
            self.model.train()
            train_loss, train_acc = 0., 0.
            
            # 记录每个样本的特征和预测
            epoch_features = {}
            epoch_predictions = {}
            epoch_softmax = {}
            
            for i, data in enumerate(train_loader):  
                # input_id, att_mask, labels, prob, pred_idx, gtlabels, group, sample_id
                input_ids, att_mask, labels, prob, pred_idx, gtlabels, group, sample_id = [Variable(elem.cuda()) for elem in data]
                
                # 前向传播
                logits = self.model(input_ids, att_mask)
                loss = loss_func(logits, labels)
                train_loss += loss.item()
                
                # 获取特征表示
                with torch.no_grad():
                    features = self.model.get_sentence_embedding(input_ids, att_mask)
                    softmax_outputs = F.softmax(logits, dim=1)
                
                # 记录预测和特征
                pred = logits.argmax(dim=-1)
                for j in range(len(sample_id)):
                    sid = sample_id[j].item()
                    epoch_features[sid] = features[j].cpu().numpy()
                    epoch_predictions[sid] = pred[j].item()
                    epoch_softmax[sid] = softmax_outputs[j].cpu().numpy()
                
                # 计算准确率
                pred_np = pred.cpu().detach().numpy()
                labels_np = labels.cpu().detach().numpy()
                train_acc += (pred_np == labels_np).sum()

                # 反向传播
                loss = loss / self.training_args.grad_acc_steps
                loss.backward()

                if (i + 1) % self.training_args.grad_acc_steps == 0 or i + 1 == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                now_samples += input_ids.size(0)
                if now_samples >= warmup_samples:
                    logger.info(" ends in %d samples", now_samples)
                    break
            
            # 记录warmup阶段的样本信息
            for i, data in enumerate(train_loader):
                input_ids, att_mask, labels, prob, pred_idx, gtlabels, group, sample_id = [elem.cuda() for elem in data]
                
                for j in range(len(sample_id)):
                    sid = sample_id[j].item()
                    if sid in epoch_features:
                        self.sample_tracking['epoch'].append(epoch_id)
                        self.sample_tracking['sample_id'].append(sid)
                        self.sample_tracking['is_labeled'].append(1)  # warmup阶段所有样本都是有标签的
                        self.sample_tracking['gmm_prob'].append(prob[j].item())
                        self.sample_tracking['loss'].append(0.0)  # warmup阶段不计算样本级别的损失
                        self.sample_tracking['pred'].append(epoch_predictions[sid])
                        self.sample_tracking['true_label'].append(gtlabels[j].item())
                        self.sample_tracking['noisy_label'].append(labels[j].item())
                        self.sample_tracking['softmax'].append(epoch_softmax[sid])
                        self.sample_tracking['features'].append(epoch_features[sid])
                        self.sample_tracking['group'].append(group[j].item())
                        
            logger.info("Warmup train samples [{:6d}/{:6d}], Loss: {:4f}, Accuracy: {:.2%}"
                        .format(now_samples, warmup_samples, train_loss / len(train_loader), train_acc / len(train_loader.dataset)))
                
            self.evaluate(train_loader, "Train")
            
            if self.eval_data is not None:
                eval_loader = self.eval_data.run("all")
                self.evaluate(eval_loader)
                
            if self.test_data is not None:
                test_loader = self.test_data.run("all")
                self.evaluate(test_loader, "Test")
                
    def evaluate(self, eval_loader, dataset_type="Eval"):
        """评估函数，记录详细的评估信息"""
        self.model.eval()

        # 创建评估记录字典
        eval_records = {
            'epoch': [],           # 当前epoch
            'sample_ids': [],      # 样本ID
            'predictions': [],     # 模型预测
            'softmax_outputs': [], # softmax输出
            'features': [],        # 特征表示
            'noisy_labels': [],    # 噪声标签
            'clean_labels': [],    # 真实标签
            'groups': [],          # 组别
            'losses': [],          # 每个样本的损失
            'correct_pred': []     # 是否预测正确
        }
        
        # 获取当前epoch
        current_epoch = len(self.training_dynamics['epoch'])
        loss_func = nn.CrossEntropyLoss(reduction='none')

        with torch.no_grad():
            for batch in eval_loader:
                input_ids, attention_mask, noisy_labels, prob, pred_idx, clean_labels, groups, sample_ids = [
                    Variable(elem.cuda()) for elem in batch
                ]
                
                # 获取模型输出
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                softmax_outputs = F.softmax(outputs, dim=1)
                
                # 计算每个样本的损失
                losses = loss_func(outputs, noisy_labels)
                
                # 获取特征表示
                features = self.model.get_sentence_embedding(input_ids, attention_mask)

                # 记录数据
                eval_records['epoch'].extend([current_epoch] * len(sample_ids))
                eval_records['sample_ids'].extend(sample_ids.cpu().numpy())
                eval_records['predictions'].extend(predictions.cpu().numpy())
                eval_records['softmax_outputs'].extend(softmax_outputs.cpu().numpy())
                eval_records['features'].extend(features.cpu().numpy())
                eval_records['noisy_labels'].extend(noisy_labels.cpu().numpy())
                eval_records['clean_labels'].extend(clean_labels.cpu().numpy())
                eval_records['groups'].extend(groups.cpu().numpy())
                eval_records['losses'].extend(losses.cpu().numpy())
                eval_records['correct_pred'].extend((predictions == clean_labels).cpu().numpy())
        
        # 将记录保存到对应的属性中
        if dataset_type.lower() == "train":
            if not hasattr(self, 'train_eval_records'):
                self.train_eval_records = {}
            self.train_eval_records[current_epoch] = eval_records
        elif dataset_type.lower() == "eval":
            if not hasattr(self, 'eval_records'):
                self.eval_records = {}
            self.eval_records[current_epoch] = eval_records
        elif dataset_type.lower() == "test":
            if not hasattr(self, 'test_records'):
                self.test_records = {}
            self.test_records[current_epoch] = eval_records
        
        # 计算并记录评估指标
        metrics = self._compute_metrics(eval_records)
        logger.info(f"dataset_type: {dataset_type}")
        logger.info(f"metrics: {metrics['clean_accuracy']}")
        logger.info(f"metrics: {metrics['noisy_accuracy']}")
        self._log_evaluation_metrics(metrics, dataset_type)
        
        return metrics['clean_accuracy']

    def _compute_metrics(self, eval_records):
        """计算评估指标"""
        predictions = np.array(eval_records['predictions'])
        clean_labels = np.array(eval_records['clean_labels'])
        noisy_labels = np.array(eval_records['noisy_labels'])
        groups = np.array(eval_records['groups'])
        
        metrics = {
            'clean_accuracy': accuracy_score(clean_labels, predictions),
            'noisy_accuracy': accuracy_score(noisy_labels, predictions),
            'group_metrics': {}
        }
        
        # 计算每个组的指标
        for group in np.unique(groups):
            group_mask = (groups == group)
            metrics['group_metrics'][group] = {
                'clean_accuracy': accuracy_score(clean_labels[group_mask], predictions[group_mask]),
                'noisy_accuracy': accuracy_score(noisy_labels[group_mask], predictions[group_mask]),
                'sample_count': np.sum(group_mask)
            }
        
        return metrics

    def _log_evaluation_metrics(self, metrics, dataset_type):
        """记录评估指标"""
        logger.info(f"\n{dataset_type} Evaluation Results:")
        logger.info(f"Clean Label Accuracy: {metrics['clean_accuracy']:.4f}")
        logger.info(f"Noisy Label Accuracy: {metrics['noisy_accuracy']:.4f}")
        
        logger.info("\nGroup-wise Results:")
        for group, group_metrics in metrics['group_metrics'].items():
            logger.info(f"Group {group}:")
            logger.info(f"  - Sample Count: {group_metrics['sample_count']}")
            logger.info(f"  - Clean Accuracy: {group_metrics['clean_accuracy']:.4f}")
            logger.info(f"  - Noisy Accuracy: {group_metrics['noisy_accuracy']:.4f}")

    def save_evaluation_records(self):
        """保存评估记录到独立的文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        # 定义保存函数
        def save_records(records, file_prefix):
            if not records:
                return
            
            file_path = os.path.join(self.save_dir, f"{file_prefix}_{timestamp}.h5")
            with h5py.File(file_path, 'w') as f:
                for epoch, epoch_data in records.items():
                    epoch_group = f.create_group(f'epoch_{epoch}')
                    for key, value in epoch_data.items():
                        if len(value) > 0:
                            epoch_group.create_dataset(
                                name=key,
                                data=np.array(value),
                                compression='gzip',
                                compression_opts=9
                            )
            logger.info(f"Saved {file_prefix} records to {file_path}")
        
        try:
            # 分别保存训练集、验证集和测试集的评估记录
            if hasattr(self, 'train_eval_records'):
                save_records(self.train_eval_records, 'train_evaluation')
            
            if hasattr(self, 'eval_records'):
                save_records(self.eval_records, 'validation_evaluation')
            
            if hasattr(self, 'test_records'):
                save_records(self.test_records, 'test_evaluation')
            
        except Exception as e:
            logger.error(f"Error saving evaluation records: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise e
    
    def train(self):
        logger.info("***** Mixup Train *****")       

        train_loader = self.train_data.run("all")
        eval_loader = self.eval_data.run("all")
        test_loader = self.test_data.run("all")
        
        # 重置样本跟踪数据，避免多次训练时数据累积
        self.sample_tracking = {
            'epoch': [],          # 所属epoch
            'sample_id': [],      # 样本ID
            'is_labeled': [],     # 是否被分到有标签集
            'gmm_prob': [],       # GMM概率值
            'loss': [],           # 损失值
            'pred': [],           # 预测标签
            'true_label': [],     # 真实标签
            'noisy_label': [],    # 噪声标签
            'softmax': [],        # softmax输出
            'features': [],       # 特征表示
            'group': []           # 组标识
        }
        
     
        for epoch_id in range(1, self.training_args.train_epochs + 1):
            logger.info("\n\n*******************   Train epoch  %d ***************************", epoch_id)
            
            start_time = time.time()
            # 先进行样本选择，计算概率
            prob, losses = self._eval_samples(train_loader)
            pred = (prob > self.method_args.p_threshold)
            logger.info(f" len(prob) is {len(prob)}")
            logger.info(f" sum(pred) is {sum(pred)}")
            logger.info(f" label data rate {sum(pred)/len(prob)}")
        
            # 创建当前epoch的样本跟踪数据
            current_epoch_tracking = {
                'epoch': [],
                'sample_id': [],
                'is_labeled': [],
                'gmm_prob': [],
                'loss': [],
                'true_label': [],
                'noisy_label': [],
                'group': []
            }
                
            # 记录每个样本的GMM概率和损失
            for i, data in enumerate(train_loader):
                input_ids, att_mask, labels, _, _, gtlabels, group, sample_id = [elem.cuda() for elem in data]
                
                for j in range(len(sample_id)):
                    sid = sample_id[j].item()
                    is_labeled = 1 if pred[sid] else 0
                    
                    current_epoch_tracking['epoch'].append(epoch_id)
                    current_epoch_tracking['sample_id'].append(sid)
                    current_epoch_tracking['is_labeled'].append(is_labeled)
                    current_epoch_tracking['gmm_prob'].append(prob[sid])
                    current_epoch_tracking['loss'].append(losses[sid])
                    current_epoch_tracking['true_label'].append(gtlabels[j].item())
                    current_epoch_tracking['noisy_label'].append(labels[j].item())
                    current_epoch_tracking['group'].append(group[j].item())
                
            # 得到有标签数据和无标签数据
            labeled_train_loader, unlabeled_train_loader = self.train_data.run("train", pred, prob)
            self.evaluate_group_distribution(labeled_train_loader, unlabeled_train_loader)
                
            # 混合训练
            train_metrics = self._train_epoch(labeled_train_loader, unlabeled_train_loader, epoch_id)
            
            # 记录训练动态
            self.training_dynamics['epoch'].append(epoch_id)
            self.training_dynamics['total_loss'].append(train_metrics['total_loss'])
            self.training_dynamics['mix_loss'].append(train_metrics['mix_loss'])
            self.training_dynamics['pse_loss'].append(train_metrics['pse_loss'])
            self.training_dynamics['kl_loss'].append(train_metrics['kl_loss'])
            self.training_dynamics['labeled_ratio'].append(sum(pred)/len(prob))
            
            # 计算有标签和无标签数据中好样本的比例
            labeled_good = 0
            labeled_total = 0
            unlabeled_good = 0
            unlabeled_total = 0
            
            for i in range(len(current_epoch_tracking['epoch'])):
                if current_epoch_tracking['is_labeled'][i] == 1:
                    labeled_total += 1
                    if current_epoch_tracking['true_label'][i] == current_epoch_tracking['noisy_label'][i]:
                        labeled_good += 1
                else:
                    unlabeled_total += 1
                    if current_epoch_tracking['true_label'][i] == current_epoch_tracking['noisy_label'][i]:
                        unlabeled_good += 1
            
            labeled_good_ratio = labeled_good / labeled_total if labeled_total > 0 else 0
            unlabeled_good_ratio = unlabeled_good / unlabeled_total if unlabeled_total > 0 else 0
            self.training_dynamics['labeled_good_ratio'].append(labeled_good_ratio)
            self.training_dynamics['unlabeled_good_ratio'].append(unlabeled_good_ratio)
            
            # 将当前epoch的跟踪数据添加到总跟踪数据中
            for key in self.sample_tracking.keys():
                if key in current_epoch_tracking:
                    self.sample_tracking[key].extend(current_epoch_tracking[key])
            
            logger.info("*** Evaluate epoch %d ***", epoch_id)
            # 评估训练集
            train_acc = self.evaluate(train_loader, "Train")
            # 评估验证集
            eval_acc = self.evaluate(eval_loader, "Eval")
            # 评估测试集
            test_acc = self.evaluate(test_loader, "Test")  
            end_time = time.time()
            
            logger.info("***  epoch cost  %.2f second ***", end_time - start_time)
            
            # 每个epoch结束后保存评估记录
            logger.info("Saving evaluation records...")
            # self.save_evaluation_records()
            
        logger.info("finish training ")
    
    
    def _train_epoch(self, labeled_train_loader, unlabeled_train_loader, epoch_id):
        labeled_train_iter = iter(labeled_train_loader)
        unlabeled_train_iter = iter(unlabeled_train_loader)
        val_iteration = len(labeled_train_loader)
        
        tenPart = val_iteration // 10
        logger.info(f"tenPart: {tenPart}")
        
        self.model.train()
        
        # 记录每个batch的数据
        batch_records = {
            'batch_idx': [],
            'epoch': [],
            # 有标签数据记录
            'labeled_samples': {
                'sample_ids': [],
                'true_labels': [],
                'noisy_labels': [],
                'features': [],
                'predictions': [],
                'softmax': [],
                'groups': []
            },
            # 无标签数据记录
            'unlabeled_samples': {
                'sample_ids': [],
                'true_labels': [],
                'noisy_labels': [],
                'features': [],
                'predictions': [],
                'softmax': [],
                'groups': []
            },
            # mixup后的数据记录
            'mixed_samples': {
                'features': [],
                'targets': [],
                'predictions': [],
                'softmax': []
            },
            # 损失值记录
            'losses': {
                'mix_loss': [],
                'pse_loss': [],
                'kl_loss': [],
                'total_loss': []
            }
        }
        
        # 用于计算平均损失
        total_loss = 0.0
        total_mix_loss = 0.0
        total_pse_loss = 0.0
        total_kl_loss = 0.0
        n_batches = 0

        # 用于计算无标签数据的准确率
        ungtLabel = []
        unprelabel = []
        
        for batch_idx in range(val_iteration):
            if batch_idx % tenPart == 0:
                self._progress_bar(
                    batch_idx, 
                    val_iteration,
                    prefix=f'Epoch {epoch_id} 训练进度',
                    suffix=f'[{batch_idx}/{val_iteration}]',
                    length=30
                )
            
            try:
                # inputs_x, inputs_x_att, targets_x, _, _, gtlabels, group, sample_ids_x = labeled_train_iter.next()
                inputs_x, inputs_x_att, targets_x, _, _, gtlabels, group, sample_ids_x = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_train_loader)
                # inputs_x, inputs_x_att, targets_x, _, _, gtlabels, group, sample_ids_x = labeled_train_iter.next()
                inputs_x, inputs_x_att, targets_x, _, _, gtlabels, group, sample_ids_x = next(labeled_train_iter)

            try:
                # inputs_u, att_u, noisy_labels_u, _, _, true_labels_u, group_u, sample_ids_u = unlabeled_train_iter.next()
                inputs_u, att_u, noisy_labels_u, _, _, true_labels_u, group_u, sample_ids_u = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_train_loader)
                # inputs_u, att_u, noisy_labels_u, _, _, true_labels_u, group_u, sample_ids_u = unlabeled_train_iter.next()
                inputs_u, att_u, noisy_labels_u, _, _, true_labels_u, group_u, sample_ids_u = next(unlabeled_train_iter)

            # 记录batch基本信息
            batch_records['batch_idx'].append(batch_idx)
            batch_records['epoch'].append(epoch_id)

            # 处理有标签数据
            targets_x = F.one_hot(targets_x, num_classes=self.model_args.num_classes)
            inputs_x, inputs_x_att, targets_x = inputs_x.cuda(), inputs_x_att.cuda(), targets_x.cuda(non_blocking=True)
            
            # 获取有标签数据的特征和预测
            with torch.no_grad():
                sents_x = self.model.get_sentence_embedding(inputs_x, inputs_x_att)
                logits_x = self.model.classify(sents_x)
                pred_x = torch.softmax(logits_x, dim=1)
                
                # 记录有标签数据信息
                batch_records['labeled_samples']['sample_ids'].extend(sample_ids_x.cpu().numpy())
                batch_records['labeled_samples']['true_labels'].extend(gtlabels.cpu().numpy())
                batch_records['labeled_samples']['noisy_labels'].extend(targets_x.argmax(dim=1).cpu().numpy())
                batch_records['labeled_samples']['features'].extend(sents_x.cpu().numpy())
                batch_records['labeled_samples']['predictions'].extend(logits_x.argmax(dim=1).cpu().numpy())
                batch_records['labeled_samples']['softmax'].extend(pred_x.cpu().numpy())
                batch_records['labeled_samples']['groups'].extend(group.cpu().numpy())

            # 处理无标签数据
            inputs_u, att_u = inputs_u.cuda(), att_u.cuda()
            
            # 获取无标签数据的特征和预测
            with torch.no_grad():
                sents_u = self.model.get_sentence_embedding(inputs_u, att_u)
                logits_u = self.model.classify(sents_u)
                sents_u2 = self.model.get_sentence_embedding(inputs_u, att_u)
                logits_u2 = self.model.classify(sents_u2)
                pred_u = torch.softmax(logits_u, dim=1)
                pred_u2 = torch.softmax(logits_u2, dim=1)
                
                # 记录无标签数据信息
                batch_records['unlabeled_samples']['sample_ids'].extend(sample_ids_u.cpu().numpy())
                batch_records['unlabeled_samples']['true_labels'].extend(true_labels_u.cpu().numpy())
                batch_records['unlabeled_samples']['noisy_labels'].extend(noisy_labels_u.cpu().numpy())
                batch_records['unlabeled_samples']['features'].extend(sents_u.cpu().numpy())
                batch_records['unlabeled_samples']['predictions'].extend(logits_u.argmax(dim=1).cpu().numpy())
                batch_records['unlabeled_samples']['softmax'].extend(pred_u.cpu().numpy())
                batch_records['unlabeled_samples']['groups'].extend(group_u.cpu().numpy())

            # mixup过程
            all_sents = torch.cat([sents_x, sents_u], dim=0)
            all_targets = torch.cat([targets_x, pred_u], dim=0)
            
            rand_idx = torch.randperm(all_sents.size(0))
            l = np.random.beta(self.method_args.alpha, self.method_args.alpha)
            l = max(l, 1 - l)
            
            mixed_sents = l * all_sents + (1 - l) * all_sents[rand_idx]
            mixed_targets = l * all_targets + (1 - l) * all_targets[rand_idx]
            
            # 获取混合后的预测
            logits = self.model.classify(mixed_sents)
            pred_mixed = torch.softmax(logits, dim=1)
            
            # 记录混合后的数据
            batch_records['mixed_samples']['features'].extend(mixed_sents.detach().cpu().numpy())
            batch_records['mixed_samples']['targets'].extend(mixed_targets.detach().cpu().numpy())
            batch_records['mixed_samples']['predictions'].extend(logits.argmax(dim=1).detach().cpu().numpy())
            batch_records['mixed_samples']['softmax'].extend(pred_mixed.detach().cpu().numpy())

            # 计算损失
            loss_mix = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * mixed_targets, dim=-1))
            pse_loss = -torch.mean(F.log_softmax(logits_u, dim=1).min(dim=1)[0])
            kl_loss = compute_kl_loss(logits_u, logits_u2)  # 使用同一批次的两次前向传播
            loss = loss_mix + pse_loss * self.method_args.lambda_p + kl_loss * self.method_args.lambda_r

            # 记录损失值
            batch_records['losses']['mix_loss'].append(loss_mix.item())
            batch_records['losses']['pse_loss'].append(pse_loss.item())
            batch_records['losses']['kl_loss'].append(kl_loss.item())
            batch_records['losses']['total_loss'].append(loss.item())

            # 累加损失
            total_mix_loss += loss_mix.item()
            total_pse_loss += pse_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
            n_batches += 1

            # 收集无标签数据的真实标签和预测
            ungtLabel.extend(true_labels_u.cpu().numpy())
            unprelabel.extend(logits_u.argmax(dim=1).cpu().numpy())

            # 反向传播
            loss = loss / self.training_args.grad_acc_steps
            loss.backward()
            
            if (batch_idx + 1) % self.training_args.grad_acc_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # 保存batch记录到HDF5文件
        # self._save_batch_records(batch_records, epoch_id)
        
        # 计算平均损失
        avg_total_loss = total_loss / n_batches
        avg_mix_loss = total_mix_loss / n_batches
        avg_pse_loss = total_pse_loss / n_batches
        avg_kl_loss = total_kl_loss / n_batches
        
        # 计算无标签数据的准确率
        unlabel_acc = accuracy_score(ungtLabel, unprelabel)
        
        # 构建并返回训练指标
        train_metrics = {
            'total_loss': avg_total_loss,
            'mix_loss': avg_mix_loss,
            'pse_loss': avg_pse_loss,
            'kl_loss': avg_kl_loss,
            'unlabel_acc': unlabel_acc
        }
        
        # 记录训练信息
        logger.info(f"Epoch {epoch_id} Training Metrics:")
        logger.info(f"Total Loss: {avg_total_loss:.4f}")
        logger.info(f"Mix Loss: {avg_mix_loss:.4f}")
        logger.info(f"Pseudo Loss: {avg_pse_loss:.4f}")
        logger.info(f"KL Loss: {avg_kl_loss:.4f}")
        logger.info(f"Unlabeled Data Accuracy: {unlabel_acc:.4f}")
        
        return train_metrics

    def _save_batch_records(self, batch_records, epoch_id):
        """保存每个epoch的batch记录"""
        save_path = os.path.join(self.save_dir, f"epoch_{epoch_id}_records.h5")
        
        with h5py.File(save_path, 'w') as f:
            # 保存基本信息
            f.create_dataset('batch_idx', data=np.array(batch_records['batch_idx']))
            f.create_dataset('epoch', data=np.array(batch_records['epoch']))
            
            # 保存有标签数据
            labeled_group = f.create_group('labeled_samples')
            for key, value in batch_records['labeled_samples'].items():
                labeled_group.create_dataset(key, data=np.array(value), compression='gzip')
            
            # 保存无标签数据
            unlabeled_group = f.create_group('unlabeled_samples')
            for key, value in batch_records['unlabeled_samples'].items():
                unlabeled_group.create_dataset(key, data=np.array(value), compression='gzip')
            
            # 保存混合数据
            mixed_group = f.create_group('mixed_samples')
            for key, value in batch_records['mixed_samples'].items():
                mixed_group.create_dataset(key, data=np.array(value), compression='gzip')
            
            # 保存损失值
            losses_group = f.create_group('losses')
            for key, value in batch_records['losses'].items():
                losses_group.create_dataset(key, data=np.array(value), compression='gzip')
        
        logger.info(f"Saved batch records for epoch {epoch_id} to {save_path}")

    def _eval_samples(self, eval_loader):
        """
        Sample selection
        """
        logger.info("sample selection: eval sample ")
        self.model.eval()
        loss_func = nn.CrossEntropyLoss(reduction='none')
        losses = np.zeros(len(eval_loader.dataset))
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                # input_ids, att_mask, labels, index,gtlabels = [Variable(elem.cuda()) for elem in data] 
                input_ids, att_mask, labels, prob, index ,gtlabels, group, sample_id = [Variable(elem.cuda()) for elem in data] 
                outputs = self.model(input_ids, att_mask) 
                pred = torch.softmax(outputs, dim=-1)
                loss = loss_func(pred, labels).cpu().detach().numpy()
                index = index.long().cpu().detach().numpy()
                losses[index] = loss
                
        # 这个正则化很重要，可能是这个关于类的正则化，使得他们的
        if self.method_args.class_reg:
            labels = np.array(eval_loader.dataset.labels, dtype=int)
            for now_class in range(self.model_args.num_classes):
                indices = np.where(labels == now_class)[0]
                losses[indices] = (losses[indices] - losses[indices].mean()) / losses[indices].var()
        else:
            losses = (losses - losses.min()) / (losses.max() - losses.min())
        
        gmm = GaussianMixture(
            n_components=2, 
            max_iter=self.method_args.gmm_max_iter, 
            tol=self.method_args.gmm_tol, 
            reg_covar=self.method_args.gmm_reg_covar
        )
        losses = losses.reshape(-1, 1)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses) 
        prob = prob[:,gmm.means_.argmin()]
        logger.info("finish select ")
        return prob,losses
    
    def save_model(self):
        self.model.save_model(self.training_args.model_save_path)

    def evaluate_group_distribution(self, labeled_loader, unlabeled_loader):
        """
        评估有标签数据和无标签数据中各组的分布情况，
        包括好样本（真实标签=噪声标签）和坏样本（真实标签≠噪声标签）的比例
        """
        # 初始化统计字典
        group_stats = {}
        
        # 总体统计
        labeled_total = 0
        labeled_good = 0  # 有标签数据中的好样本数
        unlabeled_total = 0
        unlabeled_good = 0  # 无标签数据中的好样本数
        
        # 处理有标签数据
        labeled_probs = []
        for batch in labeled_loader:
            _, _, noisy_labels, prob, _, clean_labels, groups, sample_ids = [elem.cuda() for elem in batch]
            batch_size = noisy_labels.size(0)
            labeled_total += batch_size
            
            # 计算好样本数量（噪声标签=真实标签）
            good_samples = (noisy_labels == clean_labels).sum().item()
            labeled_good += good_samples
            
            for i, group in enumerate(groups):
                group_item = group.item()
                if group_item not in group_stats:
                    group_stats[group_item] = {
                        'labeled_count': 0,
                        'labeled_good': 0,  # 有标签中的好样本
                        'labeled_bad': 0,   # 有标签中的坏样本
                        'unlabeled_count': 0,
                        'unlabeled_good': 0,  # 无标签中的好样本
                        'unlabeled_bad': 0,   # 无标签中的坏样本
                        'labeled_probs': [],
                        'unlabeled_probs': []
                    }
                
                group_stats[group_item]['labeled_count'] += 1
                group_stats[group_item]['labeled_probs'].append(prob[i].item())
                labeled_probs.append(prob[i].item())
                
                # 判断是好样本还是坏样本
                is_good = (noisy_labels[i] == clean_labels[i]).item()
                if is_good:
                    group_stats[group_item]['labeled_good'] += 1
                else:
                    group_stats[group_item]['labeled_bad'] += 1
        
        # 处理无标签数据
        unlabeled_probs = []
        for batch in unlabeled_loader:
            _, _, noisy_labels, prob, _, clean_labels, groups, sample_ids    = [elem.cuda() for elem in batch]
            batch_size = noisy_labels.size(0)
            unlabeled_total += batch_size
            
            # 计算好样本数量（噪声标签=真实标签）
            good_samples = (noisy_labels == clean_labels).sum().item()
            unlabeled_good += good_samples
            
            for i, group in enumerate(groups):
                group_item = group.item()
                if group_item not in group_stats:
                    group_stats[group_item] = {
                        'labeled_count': 0,
                        'labeled_good': 0,
                        'labeled_bad': 0,
                        'unlabeled_count': 0,
                        'unlabeled_good': 0,
                        'unlabeled_bad': 0,
                        'labeled_probs': [],
                        'unlabeled_probs': []
                    }
                
                group_stats[group_item]['unlabeled_count'] += 1
                group_stats[group_item]['unlabeled_probs'].append(prob[i].item())
                unlabeled_probs.append(prob[i].item())
                
                # 判断是好样本还是坏样本
                is_good = (noisy_labels[i] == clean_labels[i]).item()
                if is_good:
                    group_stats[group_item]['unlabeled_good'] += 1
                else:
                    group_stats[group_item]['unlabeled_bad'] += 1
        
        # 计算总样本数
        total_labeled = sum(stats['labeled_count'] for stats in group_stats.values())
        total_unlabeled = sum(stats['unlabeled_count'] for stats in group_stats.values())
        total_samples = total_labeled + total_unlabeled
        
        # 计算好样本比例
        labeled_good_ratio = labeled_good / labeled_total if labeled_total > 0 else 0
        unlabeled_good_ratio = unlabeled_good / unlabeled_total if unlabeled_total > 0 else 0
        
        # 打印总体统计信息
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"有标签样本数: {total_labeled} ({total_labeled/total_samples:.2%})")
        logger.info(f"  - 好样本数: {labeled_good} ({labeled_good_ratio:.2%})")
        logger.info(f"  - 坏样本数: {labeled_total - labeled_good} ({1-labeled_good_ratio:.2%})")
        logger.info(f"无标签样本数: {total_unlabeled} ({total_unlabeled/total_samples:.2%})")
        logger.info(f"  - 好样本数: {unlabeled_good} ({unlabeled_good_ratio:.2%})")
        logger.info(f"  - 坏样本数: {unlabeled_total - unlabeled_good} ({1-unlabeled_good_ratio:.2%})")
        
        # 计算平均概率
        avg_labeled_prob = sum(labeled_probs) / len(labeled_probs) if labeled_probs else 0
        avg_unlabeled_prob = sum(unlabeled_probs) / len(unlabeled_probs) if unlabeled_probs else 0
        logger.info(f"有标签样本平均概率: {avg_labeled_prob:.4f}")
        logger.info(f"无标签样本平均概率: {avg_unlabeled_prob:.4f}")
        
        # 打印每个组的统计信息
        logger.info("各组分布情况:")
        for group_item, stats in sorted(group_stats.items()):
            labeled_count = stats['labeled_count']
            unlabeled_count = stats['unlabeled_count']
            total_count = labeled_count + unlabeled_count
            
            # 计算比例
            labeled_ratio = labeled_count / total_labeled if total_labeled > 0 else 0
            unlabeled_ratio = unlabeled_count / total_unlabeled if total_unlabeled > 0 else 0
            
            # 计算好样本比例
            labeled_good_ratio = stats['labeled_good'] / labeled_count if labeled_count > 0 else 0
            unlabeled_good_ratio = stats['unlabeled_good'] / unlabeled_count if unlabeled_count > 0 else 0
            
            # 计算平均概率
            labeled_prob_avg = sum(stats['labeled_probs']) / len(stats['labeled_probs']) if stats['labeled_probs'] else 0
            unlabeled_prob_avg = sum(stats['unlabeled_probs']) / len(stats['unlabeled_probs']) if stats['unlabeled_probs'] else 0
            
            logger.info(f"  Group {group_item}:")
            logger.info(f"    - 总样本数: {total_count}")
            logger.info(f"    - 有标签样本: {labeled_count} ({labeled_ratio:.4f})")
            logger.info(f"      * 好样本: {stats['labeled_good']} ({labeled_good_ratio:.4f})")
            logger.info(f"      * 坏样本: {stats['labeled_bad']} ({1-labeled_good_ratio:.4f})")
            logger.info(f"      * 平均概率: {labeled_prob_avg:.4f}")
            logger.info(f"    - 无标签样本: {unlabeled_count} ({unlabeled_ratio:.4f})")
            logger.info(f"      * 好样本: {stats['unlabeled_good']} ({unlabeled_good_ratio:.4f})")
            logger.info(f"      * 坏样本: {stats['unlabeled_bad']} ({1-unlabeled_good_ratio:.4f})")
            logger.info(f"      * 平均概率: {unlabeled_prob_avg:.4f}")

    def save_tracking_data(self):
        """保存样本跟踪数据和训练动态数据到HDF5文件"""
        try:
            # 创建时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            
            # 保存样本跟踪数据
            tracking_path = os.path.join(self.save_dir, f"sample_tracking_{timestamp}.h5")
            
            # 首先确保所有数据都转换为numpy数组格式
            processed_tracking_data = {}
            for key, value in self.sample_tracking.items():
                if len(value) > 0:
                    if key in ['softmax', 'features']:
                        # 确保多维数组是numpy数组
                        processed_tracking_data[key] = np.array(value, dtype=np.float32)
                    else:
                        # 一维数组转换为numpy数组
                        processed_tracking_data[key] = np.array(value)
            
            # 处理训练动态数据
            processed_dynamics_data = {}
            for key, value in self.training_dynamics.items():
                if len(value) > 0:
                    processed_dynamics_data[key] = np.array(value, dtype=np.float32)
            
            # 处理验证集记录
            processed_eval_data = {}
            if hasattr(self, 'eval_records'):
                for epoch, records in self.eval_records.items():
                    epoch_data = {}
                    for key, value in records.items():
                        epoch_data[key] = np.array(value)
                    processed_eval_data[f'epoch_{epoch}'] = epoch_data
            
            # 处理测试集记录
            processed_test_data = {}
            if hasattr(self, 'test_records'):
                for epoch, records in self.test_records.items():
                    epoch_data = {}
                    for key, value in records.items():
                        epoch_data[key] = np.array(value)
                    processed_test_data[f'epoch_{epoch}'] = epoch_data
            
            # 在一个文件操作中完成所有数据的保存
            with h5py.File(tracking_path, 'w') as f:
                # 保存样本跟踪数据
                tracking_group = f.create_group('tracking')
                for key, value in processed_tracking_data.items():
                    if value is not None and len(value) > 0:
                        tracking_group.create_dataset(
                            name=key,
                            data=value,
                            compression='gzip',
                            compression_opts=9
                        )
                
                # 保存训练动态数据
                dynamics_group = f.create_group('dynamics')
                for key, value in processed_dynamics_data.items():
                    if value is not None and len(value) > 0:
                        dynamics_group.create_dataset(
                            name=key,
                            data=value,
                            compression='gzip',
                            compression_opts=9
                        )
                
                # 保存验证集记录
                if processed_eval_data:
                    eval_group = f.create_group('eval_records')
                    for epoch_name, epoch_data in processed_eval_data.items():
                        epoch_group = eval_group.create_group(epoch_name)
                        for key, value in epoch_data.items():
                            if value is not None and len(value) > 0:
                                epoch_group.create_dataset(
                                    name=key,
                                    data=value,
                                    compression='gzip',
                                    compression_opts=9
                                )
                
                # 保存测试集记录
                if processed_test_data:
                    test_group = f.create_group('test_records')
                    for epoch_name, epoch_data in processed_test_data.items():
                        epoch_group = test_group.create_group(epoch_name)
                        for key, value in epoch_data.items():
                            if value is not None and len(value) > 0:
                                epoch_group.create_dataset(
                                    name=key,
                                    data=value,
                                    compression='gzip',
                                    compression_opts=9
                                )
            
            logger.info(f"Successfully saved tracking data to {tracking_path}")
            
            # 保存元数据
            meta_path = os.path.join(self.save_dir, f"metadata_{timestamp}.txt")
            with open(meta_path, 'w') as f:
                f.write(f"Training Arguments:\n")
                for arg, value in vars(self.training_args).items():
                    f.write(f"{arg}: {value}\n")
                f.write(f"\nMethod Arguments:\n")
                for arg, value in vars(self.method_args).items():
                    f.write(f"{arg}: {value}\n")
                f.write(f"\nModel Arguments:\n")
                for arg, value in vars(self.model_args).items():
                    f.write(f"{arg}: {value}\n")
                
            logger.info(f"Successfully saved metadata to {meta_path}")
            
        except Exception as e:
            logger.error(f"Error saving tracking data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise e

    def _ensure_numpy(self, data):
        """确保数据是numpy数组格式"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            if len(data) == 0:
                return np.array([])
            # 处理列表中的元素
            first_elem = data[0]
            if isinstance(first_elem, torch.Tensor):
                return np.array([t.cpu().numpy() for t in data])
            else:
                return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def _progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
        """
        创建一个日志进度条
        Parameters:
            iteration  - 当前迭代次数
            total      - 总迭代次数
            prefix    - 前缀字符串
            suffix    - 后缀字符串
            decimals  - 进度百分比的小数位数
            length    - 进度条的字符长度
            fill      - 进度条填充字符
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        progress_line = f"{prefix} |{bar}| {percent}% {suffix}"
        logger.info(progress_line)
