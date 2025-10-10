import yaml
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


def load_yaml_to_dataclass(yaml_path, dataclass_types):
    """
    从YAML文件加载配置到dataclass
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results = []
    # 从method部分获取方法名
    
    
    for dataclass_type in dataclass_types:
        instance = dataclass_type()
        fields = asdict(instance)
        
        if dataclass_type.__name__ == 'ModelArguments':
            # 只处理model部分的参数
            model_args = config.get('model', {})
            for k, v in model_args.items():
                if k in fields:
                    setattr(instance, k, v)
                    
        elif dataclass_type.__name__ == 'DataTrainingArguments':
            # 处理data部分的参数
            data_args = config.get('data', {})
            
            # 获取数据模式和类型
            data_mode = data_args.get('dataMode', 'test')
            data_type = data_args.get('dataType', 'Worst')
            
            # 根据模式选择对应的路径配置
            paths_key = 'test_mode_paths' if data_mode == 'test' else 'training_mode_paths'
            if paths_key in data_args and data_type in data_args[paths_key]:
                path_config = data_args[paths_key][data_type]
                # 设置文件路径
                data_args['train_file_path'] = path_config['train_file']
                data_args['eval_file_path'] = path_config['eval_file']
                data_args['test_file_path'] = path_config['test_file']
            
            # 设置其他参数
            for k, v in data_args.items():
                if k in fields:
                    setattr(instance, k, v)
                    
        elif dataclass_type.__name__ == 'OurTrainingArguments':
            # 处理training部分的参数
            training_args = config.get('training', {})
            for k, v in training_args.items():
                if k in fields:
                    setattr(instance, k, v)
                    
        elif dataclass_type.__name__ == 'MethodArguments':
            # 获取method_name和对应方法的参数
            method_config = config.get('method', {})
            method_name = method_config.get('method_name', 'selfmix')
            method_args = method_config.get(method_name, {})
            
            # 将method_name添加到method_args中
            method_args['method_name'] = method_name
            
            # 设置所有参数
            for k, v in method_args.items():
                if k in fields:
                    setattr(instance, k, v)
            print(f"Method arguments: {method_args}")
        
        results.append(instance)
    
    return tuple(results)