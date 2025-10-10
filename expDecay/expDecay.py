from dataclasses import dataclass, field
from typing import Optional
import sys, os
from .expDecayDataset import *   
from .expDecayModel import expDecayModel
from .expDecayTrainer import expDecayTrainer

from transformers import (
    AutoModel,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
)
import time 


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
logger = logging.getLogger(__name__)


    
def trainExpDecay(model_args, data_args, training_args, method_args):
    
        
    start_time = time.time()
    
   
    # load data
    train_datasets, train_num_classes = load_dataset(data_args.train_file_path, data_args.dataset_name)
    eval_datasets, eval_num_classes = load_dataset(data_args.eval_file_path, data_args.dataset_name)
    test_datasets, test_num_classes = load_dataset(data_args.test_file_path, data_args.dataset_name)
    # 将数据集分成训练集和验证集
    assert train_num_classes == eval_num_classes
    
    model_args.num_classes = train_num_classes
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
    expdecay_train_data = expDecayData(data_args, train_datasets, tokenizer)
    expdecay_eval_data = expDecayData(data_args, eval_datasets, tokenizer)  
    expdecay_test_data = expDecayData(data_args, test_datasets, tokenizer)
    # load model
    model = expDecayModel(model_args.pretrained_model_name_or_path, model_args.dropout_rate, model_args.num_classes)
    
    # build trainer
    trainer = expDecayTrainer(
        model=model,
        train_data=expdecay_train_data,  
        eval_data=expdecay_eval_data,
        test_data=expdecay_test_data,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        method_args=method_args
    )
    
    # train and eval
    test_best_l, test_last_l = trainer.dynamic_train()
   
    end_time = time.time()
    
    logger.info(f"训练的持续时间为 {end_time - start_time}")


