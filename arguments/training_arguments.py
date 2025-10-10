from dataclasses import dataclass, field
from typing import Optional

@dataclass
class OurTrainingArguments:
    """
    训练过程相关的参数配置
    """
    seed: Optional[int] = field(
        default=1,
        metadata={"help": "Random seed"}
    )
    
    train_epochs: int = field(
        default=2,
        metadata={"help": "Training epochs"}
    )
    
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )
    
    grad_acc_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps"}
    )
    
    model_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save model"}
    )
    
    warmup_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": "Warmup strategy: [no, epoch, samples]"
        }
    )
    
    warmup_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of epochs to warmup the model"
        }
    )
    
    warmup_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of samples to warmup the model"
        }
    )
    split_num: Optional[int] = field(
        default=20,
        metadata={
            "help": "eval step split"
        }
    )
   
    optimizer: Optional[str] = field(
        default='adam',
        metadata={
            "help": "Optimizer: [adam, sgd]"
        }
    )
    
    # LNPL方法的参数
    # optimizer: Optional[str] = field(
    #     default='adam',
    #     metadata={
    #         "help": "Optimizer: [adam, sgd]"
    #     }
    # )
    # # 优化器的参数
    # initializer: Optional[str] = field(
    #     default='xavier_uniform_',
    #     metadata={
    #         "help": "Initializer: [xavier, kaiming]"
    #     }
    # )
    # num_epoch_negative: int = field(
    #     default=3,
    #     metadata={
    #         "help": "epochs for negative training"
    #     }
    # )
    # l2reg: float = field(
    #     default=0.01,
    #     metadata={
    #         "help": "L2 regularization"
    #     }
    # )
    # switch_epoch: int = field(
    #     default=7,
    #     metadata={
    #         "help": "switch selective mode"
    #     }
    # )
    # num_hist: int = field(
    #     default=2,
    #     metadata={
    #         "help": "number of epoch to save histogram"
    #     }
    # )
    # neg_sample_num: int = field(
    #     default=10,
    #     metadata={
    #         "help": "try larger number for non-BERT models"
    #     }
    # )
    # warmup_proportion: float = field(
    #     default=0.002,
    #     metadata={
    #         "help": "warmup proportion"
    #     }
    # )
    
    # scenario: int = field(
    #     default=1,
    #     metadata={
    #         "help": "1 for truth noise 2 filtter by loss"
    #     }
    # )
    # symm: int = field(
    #     default=1,
    #     metadata={
    #         "help": "1 for truth noise 2 filtter by loss"
    #     }
    # )
    # noise_percentage: float = field(
    #     default=0.0,
    #     metadata={
    #         "help": "0 false or 1 true"
    #     }
    # )
    # lebel_dim: int = field(
    #     default=2,
    #     metadata={
    #         "help": "label dimension"
    #     }
    # )
    # plm: str = field(
    #     default='bert',
    #     metadata={
    #         "help": "pretrain molel"
    #     }
    # )
    # save_model_nt: int = field(
    #     default=0,
    #     metadata={
    #         "help": "0 false or 1 true"
    #     }
    # )
    # use_ads: int = field(
    #     default=1,
    #     metadata={
    #         "help": "0 false or 1 true"
    #     }
    # )
    # use_relabel: int = field(
    #     default=1,
    #     metadata={
    #         "help": "0 false or 1 true"
    #     }
    # )
    # save_dir: str = field(
    #     default='state_dict',
    #     metadata={
    #         "help": "save directory"
    #     }
    # )
    
    
    
    # max_seq_len: int = field(
    #     default=50,
    #     metadata={
    #         "help": "The maximum sequence length"
    #     }
    #  )  被训练参数max_sentence_len替代

    
    
    

    

    