from .base_config import ModelConfig

#custom model config
UNET_CONFIG = ModelConfig(
    model_type = 'unet',
    dropout_rate = 0.3, #aids with overfitting / regularization
    use_batch_norm = True
)

ATTENTION_UNET_CONFIG = ModelConfig(
    model_type='attention_unet',
    dropout_rate = 0.3,
    use_batch_norm = True
)

EFFICIENTNET_B8_CONFIG = ModelConfig(
    model_type = 'efficientnet',
    backbone = 'efficientnetb0',
    pretrained_weights = 'imagenet',
    freeze_backbone = False,
    dropout_rate = 0.3
)

RESNET50_CONFIG = ModelConfig(
    model_type = 'resnet',
    backbone = 'resnet50',
    pretrained_weights = 'imagenet',
    freeze_backbone = False,
    dropout_rate = 0.3
)

MOBILENET_V2_CONFIG = ModelConfig(
    model_type = 'mobilenet',
    backbone = 'mobilenetv2',
    pretrained_weights = 'imagenet',
    freeze_backbone = False,
    dropout_rate = 0.3
)


#model dict
MODEL_CONFIGS = {
    'unet' : UNET_CONFIG,
    'attention_unet' : ATTENTION_UNET_CONFIG,
    'efficientnet_b0' : EFFICIENTNET_B8_CONFIG,
    'resnet50' : RESNET50_CONFIG,
    'mobilenet_v2' : MOBILENET_V2_CONFIG,
}