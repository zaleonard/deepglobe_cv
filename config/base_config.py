#config for all expirments

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
import yaml

@dataclass
class DataConfig:
    train_images_path: str = "training_images.pkl"
    train_masks_path: str = "training_masks.pkl"
    test_images_path: str = "test_images.pkl"
    test_masks_path: str = "test_masks.pkl"
    num_classes: int = ...
    img_height: int = ...
    img_width: int = ...
    val_split: float = 0.2
    augmentation: bool = True


@dataclass
class ModelConfig:
    model_type: str = "unet" #unet options go here
    backbone: str = None #transfer learning stuff
    pretrained_weights: str = None #pre-trained weights path
    freeze_backbone: bool = False
    dropout_rate: float = 0.3
    use_batch_norm: bool = True

@dataclass
class TrainingConfig:
    #hyperparameter training
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float  = 1e-4
    optimizer: str = "adam" #adam, sgd, adamw 
    loss_function: str = "categorical_crossentropy" #or custom
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    class_weights: Dict[int, float] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    #experiment configurations
    experiment_name: str = "baseline_exp"
    random_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_facotry=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Path):
        #save config
        path.parent.mkdir(parents=True, exists_ok=True)

        #convert to dict
        config_dict = {
            'experiment_name' : self.experiment_name,
            'random_seed': self.random_seed,
            'data' : self.data.__dict__,
            'model' : self.model.__dict__,
            'training' : self.training.__dict__
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path):
        #load yaml config
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            experiment_name = config_dict['experiment_name'],
            random_seed = config_dict['random_seed'],
            data = DataConfig(**config_dict['data']),
            model = ModelConfig(**config_dict['model']),
            training = TrainingConfig(**config_dict['training'])
        )