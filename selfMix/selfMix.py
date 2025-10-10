from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import logging, sys, os, random
import torch
import numpy as np
from .selfMixDataset import *
from .selfMixModel import *
from .selfMixTrainer import *

from transformers import (
    AutoModel,
    AutoTokenizer,
)
import time 


logger = logging.getLogger(__name__)

    
def train_selfMix(model_args, data_args, training_args, method_args):
    start_time = time.time()
    
    # load data
    train_datasets, train_num_classes = load_dataset(data_args.train_file_path, data_args.dataset_name)
    eval_datasets, eval_num_classes = load_dataset(data_args.eval_file_path, data_args.dataset_name)
    test_datasets, test_num_classes = load_dataset(data_args.test_file_path, data_args.dataset_name)
    assert train_num_classes == eval_num_classes
    model_args.num_classes = train_num_classes
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
    selfmix_train_data = SelfMixData(data_args, train_datasets, tokenizer)
    selfmix_eval_data = SelfMixData(data_args, eval_datasets, tokenizer)
    selfmix_test_data = SelfMixData(data_args, test_datasets, tokenizer)
    
    # load model
    model = Bert4Classify(model_args.pretrained_model_name_or_path, model_args.dropout_rate, model_args.num_classes)
    
    # build trainer
    trainer = SelfMixTrainer(
        model=model,
        train_data=selfmix_train_data,
        eval_data=selfmix_eval_data,
        test_data=selfmix_test_data,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        method_args=method_args
    )
    
    # train and eval
    trainer.warmup()
    trainer.train()
    # trainer.save_model()
    end_time = time.time()
    logger.info(f"Total training time: {end_time - start_time} seconds")