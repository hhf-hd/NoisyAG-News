from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    """
    数据处理相关的参数配置
    """
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of dataset"}
    )
    
    train_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The train data file (.csv)"}
    )
    
    eval_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The eval data file (.csv)"}
    )
    
    test_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The test data file (.csv)"}
    )
    
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size"}
    )
    
    batch_size_mix: int = field(
        default=16,
        metadata={"help": "Batch size for mix train"}
    )
    
    batch_size_val: int = field(
        default=32,
        metadata={
            "help": "batch size for inference"
        }
    )
    
    max_sentence_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sentence length after tokenization."
        }
    )
    
    
    dataMode: Optional[str] = field(
        default="test",
        metadata={
            "help": "The mode of data"
        }
    )
    
    dataType: Optional[str] = field(
        default="Worst",
        metadata={
            "help": "The type of data"
        }
    )
    
    saveTrainInfo: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to save the train info"
        }
    )
    
    
    