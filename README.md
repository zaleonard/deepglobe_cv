# AI Gating Project - Multi-Class Segmentation

A modern, modular deep learning project for multi-class segmentation of flow cytometry gating data. Supports both custom and pre-trained models with comprehensive experiment tracking.

## Project Overview

This project implements a computer vision pipeline to differentiate between 4 different classes in flow cytometry dot plots:
- **urban_land** (Class 0)
- **agriculture_land** (Class 1)  
- **rangeland** (Class 2)
- **forest_land** (Class 3)
- **water** (Class 4)
- **barren_land** (Class 5)
- **unknown** (Class 6)

The architecture supports:
-  Multiple model architectures (U-Net, Attention U-Net, EfficientNet, MobileNet)
-  Transfer learning with pre-trained backbones
-  Comprehensive experiment tracking with TensorBoard
-  Multiple loss functions (Cross-Entropy, Dice, Focal, Combined)
-  Data augmentation pipeline
-  Per-class performance metrics

## Project Structure

```
ai_gating_project/
│
├── config/                  # Configuration management
│   ├── base_config.py      # Core configuration classes
│   ├── model_configs.py    # Predefined model configs
│   └── training_configs.py # Training hyperparameter presets
│
├── data/                   # Data loading and preprocessing
│   ├── dataset.py         # Custom dataset class
│   ├── augmentation.py    # Data augmentation pipeline
│   └── preprocessing.py   # Image/mask preprocessing utils
│
├── models/                # Model architectures
│   ├── base_model.py     # Abstract base class
│   ├── custom/           # Custom architectures
│   │   └── unet.py      # U-Net and Attention U-Net
│   └── pretrained/      # Transfer learning models
│       ├── efficientnet.py
│       └── mobilenet.py
│
├── training/             # Training orchestration
│   ├── trainer.py       # Main training loop
│   ├── losses.py        # Custom loss functions
│   └── callbacks.py     # Training callbacks
│
├── evaluation/          # Model evaluation
│   ├── metrics.py      # Custom metrics (IoU, Dice, etc.)
│   └── evaluator.py    # Comprehensive evaluation
│
├── utils/              # Utilities
│   ├── logger.py      # Logging setup
│   ├── visualization.py # Plotting functions
│   └── io.py          # I/O operations
│
├── experiments/        # Training outputs (auto-generated)
├── notebooks/         # Jupyter notebooks for analysis
│
├── train.py          # Main training script
├── evaluate.py       # Model evaluation script
├── predict.py        # Inference script
└── requirements.txt  # Python dependencies
```

##  Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure you have your training data in folders:
 - train
    - images
    - masks
 - val
    - images
 - test
    - images

### 3. Train a Model

#### Basic Training (U-Net)
```bash
python train.py --experiment-name unet_baseline --model-type unet --epochs 50 --batch-size 8
```

#### Transfer Learning (EfficientNet)
```bash
python train.py --experiment-name efficientnet_transfer \
                --model-type efficientnet \
                --backbone efficientnetb0 \
                --epochs 30 \
                --batch-size 16 \
                --augment
```

#### Advanced Training with Custom Loss
```bash
python train.py --experiment-name unet_dice_loss \
                --model-type attention_unet \
                --loss dice \
                --learning-rate 1e-3 \
                --epochs 100 \
                --augment \
                --dropout 0.4
```

### 4. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=experiments

# Open browser to http://localhost:6006
```

### 5. Evaluate Model

```bash
python evaluate.py --model-path experiments/unet_baseline_20260105_123456/checkpoints/best_model.keras \
                   --test-images test_images.pkl \
                   --test-masks test_masks.pkl \
                   --output-dir evaluation_results
```

### 6. Run Inference

```bash
python predict.py --model-path experiments/unet_baseline_20260105_123456/checkpoints/best_model.keras \
                  --input-path new_images.pkl \
                  --output-dir predictions \
                  --visualize
```

##  Command-Line Options

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--experiment-name` | Name of experiment | `baseline` |
| `--model-type` | Model architecture | `unet` |
| `--backbone` | Pretrained backbone | `None` |
| `--epochs` | Number of epochs | `50` |
| `--batch-size` | Batch size | `8` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--optimizer` | Optimizer (adam/sgd/adamw) | `adam` |
| `--loss` | Loss function | `categorical_crossentropy` |
| `--augment` | Enable data augmentation | `False` |
| `--dropout` | Dropout rate | `0.3` |
| `--val-split` | Validation split ratio | `0.2` |

## 🏗️ Model Architectures

### Available Models

1. **U-Net** (`--model-type unet`)
   - Classic encoder-decoder architecture
   - Skip connections for fine-grained segmentation
   - Best for: General purpose segmentation

2. **Attention U-Net** (`--model-type attention_unet`)
   - U-Net with attention gates
   - Focuses on relevant features
   - Best for: Complex segmentation tasks

3. **EfficientNet** (`--model-type efficientnet --backbone efficientnetb0`)
   - Transfer learning with EfficientNet backbone
   - Variants: efficientnetb0 through efficientnetb7
   - Best for: When you have limited data

4. **MobileNet** (`--model-type mobilenet --backbone mobilenetv2`)
   - Lightweight architecture
   - Fast inference
   - Best for: Deployment on resource-constrained devices

## 📊 Loss Functions

| Loss Function | Use Case | Command |
|---------------|----------|---------|
| **Categorical Cross-Entropy** | Balanced datasets | `--loss categorical_crossentropy` |
| **Dice Loss** | Imbalanced classes, overlap important | `--loss dice` |
| **Focal Loss** | Severe class imbalance | `--loss focal` |
| **Combined Loss** | Dice + Cross-Entropy | `--loss combined` |

## 📈 Experiment Tracking

Every training run creates an experiment folder:

```
experiments/unet_baseline_20260105_123456/
├── config.yaml                 # Full configuration
├── checkpoints/
│   ├── best_model.keras       # Best model (lowest val_loss)
│   └── final_model.keras      # Final epoch model
├── logs/                      # TensorBoard logs
└── results/
    ├── training_log.csv       # Epoch-by-epoch metrics
    ├── history.json           # Complete history
    └── experiment_summary.json # Summary
```

## 🔍 Evaluation Metrics

The evaluation script computes:

- **Pixel Accuracy**: Overall correct pixels
- **Mean IoU**: Average Intersection over Union
- **Dice Coefficient**: F1 score for segmentation
- **Per-Class Metrics**: IoU, Dice, Accuracy for each class
- **Confusion Matrix**: Detailed class confusion

## Usage Examples

### Compare Multiple Models

```bash
# Train U-Net
python train.py --experiment-name unet_exp --model-type unet --epochs 50

# Train Attention U-Net
python train.py --experiment-name attention_unet_exp --model-type attention_unet --epochs 50

# Train EfficientNet
python train.py --experiment-name efficientnet_exp --model-type efficientnet --backbone efficientnetb0 --epochs 30

# View all experiments in TensorBoard
tensorboard --logdir=experiments
```

### Fine-tune Pre-trained Model

```bash
# Step 1: Train with frozen backbone
python train.py --experiment-name efficientnet_frozen \
                --model-type efficientnet \
                --backbone efficientnetb0 \
                --freeze-backbone \
                --epochs 20 \
                --learning-rate 1e-3

# Step 2: Fine-tune entire model
python train.py --experiment-name efficientnet_finetuned \
                --model-type efficientnet \
                --backbone efficientnetb0 \
                --epochs 30 \
                --learning-rate 1e-5
```

### Hyperparameter Search

```bash
# Different learning rates
for lr in 1e-3 1e-4 1e-5; do
    python train.py --experiment-name unet_lr_${lr} --learning-rate ${lr}
done

# Different batch sizes
for bs in 4 8 16; do
    python train.py --experiment-name unet_bs_${bs} --batch-size ${bs}
done
```

## Custom Development

### Adding a New Model

1. Create new model class in `models/custom/` or `models/pretrained/`
2. Inherit from `BaseSegmentationModel`
3. Implement `build_model()` method
4. Register in `train.py` `get_model_class()` function

Example:
```python
# models/custom/my_model.py
from models.base_model import BaseSegmentationModel

class MyModel(BaseSegmentationModel):
    def build_model(self):
        # Build your architecture
        inputs = layers.Input(shape=self.input_shape)
        # ... your layers ...
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(x)
        return Model(inputs, outputs)
```

### Adding a New Loss Function

Add to `training/losses.py`:
```python
def my_custom_loss(y_true, y_pred):
    # Your loss implementation
    return loss_value
```

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Use smaller model (`--model-type unet` instead of `efficientnet`)
- Reduce `--img-size`

### Poor Performance
- Enable `--augment`
- Try different `--loss` functions (especially `dice` for imbalanced data)
- Increase `--epochs`
- Try transfer learning (`--model-type efficientnet --backbone efficientnetb0`)

### Training Too Slow
- Increase `--batch-size` (if memory allows)
- Use `--model-type mobilenet` for faster training
- Reduce model depth

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Attention U-Net Paper](https://arxiv.org/abs/1804.03999)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)



## Contributing

Feel free to extend this project with:
- New model architectures
- Additional loss functions
- Advanced data augmentation techniques
- Deployment scripts (TFLite, ONNX)

---

