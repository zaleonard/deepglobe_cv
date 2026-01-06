from .base_config import TrainingConfig

#configurations for terminal quick tests

QUICK_TEST_CONFIG  =TrainingConfig(
    epochs = 5,
    batch_size = 4,
    learning_rate = 1e-3,
    optimizer = "adam",
    early_stoppping_patience = 3,
    reduce_lr_patience = 2,
)

STANDARD_CONFIG  =TrainingConfig(
    epochs = 20,
    batch_size = 8,
    learning_rate = 1e-4,
    optimizer = "adam",
    early_stoppping_patience = 10,
    reduce_lr_patience = 5,
)

EXTENDED_CONFIG  =TrainingConfig(
    epochs = 100,
    batch_size = 8,
    learning_rate = 1e-4,
    optimizer = "adam",
    early_stoppping_patience = 10,
    reduce_lr_patience = 5,
)

FINETUNE_CONFIG  =TrainingConfig(
    epochs = 30,
    batch_size = 8,
    learning_rate = 1e-5,
    optimizer = "adam",
    early_stoppping_patience = 8,
    reduce_lr_patience = 4,
)

HIGH_LR_CONFIG  =TrainingConfig(
    epochs = 50,
    batch_size = 8,
    learning_rate = 5e-4,
    optimizer = "adam",
    early_stoppping_patience = 10,
    reduce_lr_patience = 5,
)

#training dict

TRAINING_CONFIGS = {
    'quick_test' : QUICK_TEST_CONFIG,
    'standard' : STANDARD_CONFIG,
    'extended' : EXTENDED_CONFIG,
    'finetune' : FINETUNE_CONFIG,
    'high_lr' : HIGH_LR_CONFIG
}