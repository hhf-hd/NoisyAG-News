from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MethodArguments:
    """
    不同方法的特定参数配置
    """
    # 当前使用的方法名
    method_name: str = field(
        default='selfmix',
        metadata={"help": "当前使用的方法名: [WN，LS, NLS, CT, selfMix, expDecay]"}
    )

    # LS方法的参数
    label_smoothing: Optional[float] = field(
        default=None,
        metadata={"help": "Label smoothing factor for LS method"}
    )

    # NLS方法的参数
    negative_smoothing: Optional[float] = field(
        default=None,
        metadata={"help": "Negative label smoothing factor for NLS method"}
    )
    
    # CT方法参数    
    forget_rate: Optional[float] = field(
        default=None,
        metadata={"help": "CT Forget rate parameter"}
    )
    model1_name: Optional[str] = field(
        default=None,
        metadata={"help": "CT Model1 name parameter"}
    )
    model2_name: Optional[str] = field(
        default=None,
        metadata={"help": "CT Model2 name parameter"}
    )
    model1_optimizer: Optional[str] = field(
        default=None,
        metadata={"help": "CT Model1 optimizer parameter"}
    )
    model2_optimizer: Optional[str] = field(
        default=None,
        metadata={"help": "CT Model2 optimizer parameter"}
    )   
    
    
    # expDecay方法的参数
    exp: Optional[float] = field(
        default=None,
        metadata={"help": "Exponent factor for expDecay method"}
    )
    p_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Probability threshold for expDecay method"}
    )
    temp: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature parameter"}
    )

    # selfMix方法的参数
    alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Alpha parameter for selfMix method"}
    )
    class_reg: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use class regularization"}
    )
    gmm_max_iter: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum iterations for GMM"}
    )
    gmm_reg_covar: Optional[float] = field(
        default=None,
        metadata={"help": "Regularization parameter for GMM covariance"}
    )
    gmm_tol: Optional[float] = field(
        default=None,
        metadata={"help": "Tolerance for GMM convergence"}
    )
    lambda_p: Optional[float] = field(
        default=None,
        metadata={"help": "Lambda p parameter"}
    )
    lambda_r: Optional[float] = field(
        default=None,
        metadata={"help": "Lambda r parameter"}
    )
    p_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "P threshold parameter"}
    )
    temp: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature parameter"}
    )
    
    
    
    def __str__(self):
        """自定义字符串表示，只显示非None的值"""
        fields = []
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                fields.append(f"{field_name}={field_value}")
        return f"MethodArguments({', '.join(fields)})" 