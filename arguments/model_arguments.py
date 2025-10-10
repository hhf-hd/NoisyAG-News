from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    模型相关的参数配置
    """
    
    pretrained_model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={
            "help": "The pretrained model checkpoint for weights initialization."
        }
    )
    
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"}
    )
    
    num_classes: Optional[int] = field(
        default=4,
        metadata={"help": "Number of classes for classification"}
    )
    
    device: str = field(
        default='cuda:0',
        metadata={
            "help": "Device to run the model on. e.g. cuda:0"
        }
    )
    
    gpuid: str = field(
        default='0',
        metadata={
            "help": "GPU ID to run the model on. e.g. 0"
        }
    )
    
    
    