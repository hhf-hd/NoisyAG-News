import logging
import sys
import os
import torch
import re
from transformers import HfArgumentParser, set_seed
from utils.mydataloader import create_dataloader
from arguments.model_arguments import ModelArguments
from arguments.data_arguments import DataTrainingArguments
from arguments.training_arguments import OurTrainingArguments
from arguments.method_arguments import MethodArguments
from WN.baseline import train_baseline
from selfMix.selfMix import train_selfMix
from DenoMix.DenoMix import train_DenoMix
from LS.LS import train_LS
from NLS.NLS import train_NLS
from utils.yaml_loader import load_yaml_to_dataclass
# from utils.unzipBert import unzipBert
from expDecay.expDecay import trainExpDecay
from CT.CT import trainCT
# from LNPL.LNPL import trainLNPL
import time 

logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments, MethodArguments))
    
    if len(sys.argv) == 2:
        if sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml"):
            model_args, data_args, training_args, method_args = load_yaml_to_dataclass(
                yaml_path=os.path.abspath(sys.argv[1]),
                dataclass_types=(ModelArguments, DataTrainingArguments, OurTrainingArguments, MethodArguments)
            )
    else:
        model_args, data_args, training_args, method_args = parser.parse_args_into_dataclasses()
    
    data_feature = "Unknown"
    if data_args.train_file_path:
        match = re.search(r'([A-Z][a-z]*)\.csv$', data_args.train_file_path)
        if match:
            data_feature = match.group(1)
    data_feature = data_args.dataType
    
    method_name = method_args.method_name.split('-')[0]  
    if method_name == "selfMix":
        if method_args.class_reg:
            log_filename = f"{method_name}-{data_feature}-{training_args.seed}-{method_args.alpha}-{method_args.temp}-{method_args.p_threshold}-{method_args.lambda_p}-{method_args.lambda_r}-classReg.log"
        else:
            log_filename = f"{method_name}-{data_feature}-{training_args.seed}-{method_args.alpha}-{method_args.temp}-{method_args.p_threshold}-{method_args.lambda_p}-{method_args.lambda_r}.log"
    elif method_name == "expDecay":
        log_filename = f"{method_name}-{data_feature}-s{training_args.seed}-exp{method_args.exp}-lr{method_args.lambda_r}-ph{method_args.p_threshold}-T{method_args.temp}-dp{model_args.dropout_rate}.log"
    elif method_name == "WN":
        log_filename = f"{method_name}-{data_feature}-s{training_args.seed}-bs{data_args.batch_size}.log"
    elif method_name == "LS":
        log_filename = f"{method_name}-{data_feature}-s{training_args.seed}-ls{method_args.label_smoothing}-bs{data_args.batch_size}"
    elif method_name == "NLS":
        log_filename = f"{method_name}-{data_feature}-s{training_args.seed}-nls{data_args.negative_smoothing}-bs{data_args.batch_size}"
    elif method_name == "CT":
        log_filename = f"{method_name}-{data_feature}-s{training_args.seed}-bs{data_args.batch_size}-fr{method_args.forget_rate}-m1{method_args.model1_name}-m2{method_args.model2_name}-m1o{method_args.model1_optimizer}-m2o{method_args.model2_optimizer}.log"
    
    else:
        log_filename = f"{method_name}-{data_feature}-{training_args.seed}.log"
    
    # 创建日志目录
    os.makedirs('./logs', exist_ok=True)
    os.makedirs(f'./logs/{method_name}', exist_ok=True)
    log_path = os.path.join(f'./logs/{method_name}', log_filename) 
    
    # 配置日志 - 只使用 handlers 参数，不使用 filename 参数
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练...")
    
    logger.info(f"日志文件: {log_path}")
    logger.info(f"使用方法: {method_args.method_name}")
    logger.info(f"数据特征: {data_feature}")
    
    # data_args.dataType = data_feature
    
    logger.info("Model Parameters %s", model_args)
    for key, value in model_args.__dict__.items():
        logger.info(f"{key}: {value}")
    logger.info("Data Parameters %s", data_args)
    for key, value in data_args.__dict__.items():
        logger.info(f"{key}: {value}")
    logger.info("Training Parameters %s", training_args)
    for key, value in training_args.__dict__.items():
        logger.info(f"{key}: {value}")
    logger.info("Method Parameters %s", method_args)
    for key, value in method_args.__dict__.items():
        if value is not None:  # 只打印非None的值
            logger.info(f"{key}: {value}")
    
    device = torch.device("cuda:" + str(model_args.gpuid) if torch.cuda.is_available() else "cpu")
    model_args.device = device
    
    # 把环境都固定住
    set_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    # 创建数据加载器,返回的是 
    # ['input_ids', 'attention_mask', 'noisy_label', 'clean_label', 'group', 'text','sample_id']
    train_dataloader, _ = create_dataloader(
        file_path=data_args.train_file_path,
        tokenizer_path=model_args.pretrained_model_name_or_path,
        batch_size=data_args.batch_size,
        max_length=data_args.max_sentence_len,
        shuffle=True
    )
    

    eval_dataloader, _ = create_dataloader(
        file_path=data_args.eval_file_path,
        tokenizer_path=model_args.pretrained_model_name_or_path,
        batch_size=data_args.batch_size,
        max_length=data_args.max_sentence_len,
        shuffle=False
    )
  
    test_dataloader, _ = create_dataloader(
        file_path=data_args.test_file_path,
        tokenizer_path=model_args.pretrained_model_name_or_path,
        batch_size=data_args.batch_size,
        max_length=data_args.max_sentence_len,  
        shuffle=False
    )
    
    if method_args.method_name.startswith("WN"):
        train_baseline(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            test_dataloader=test_dataloader,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            method_args=method_args
        )
    elif method_args.method_name.startswith("LS"):
        train_LS(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            test_dataloader=test_dataloader,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            method_args=method_args
        )
    elif method_args.method_name.startswith("NLS"):
        train_NLS(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            test_dataloader=test_dataloader,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            method_args=method_args
        )
    elif method_args.method_name.startswith("CT"):
        trainCT(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            method_args=method_args
        )
    elif method_args.method_name.startswith("selfMix"):
        train_selfMix(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            method_args=method_args,
        )
    
    elif method_args.method_name.startswith("expDecay"):
        trainExpDecay(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            method_args=method_args,
        )
        
    elif method_args.method_name.startswith("DenoMix"):
        train_DenoMix(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            method_args=method_args,
        )
    
    else:
        logger.info(f"Method {method_args.method_name} is not supported in this script.")
    

    end_time = time.time()
    logger.info(f"训练结束，用时{end_time - start_time}秒")

if __name__ == "__main__":
    main()
    
    


    


    


