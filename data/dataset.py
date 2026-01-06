import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import kagglehub
import os
import shutil
from PIL import Image

def resize_images_in_folder(input_folder: str, output_folder: str, size: Tuple[int, int] = (256, 256)):
    """
    Resize all images in a folder to specified dimensions.
    
    Args:
        input_folder: Path to folder containing images
        output_folder: Path to save resized images
        size: Target size as (width, height). Default is (256, 256)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in supported_formats]
    
    print(f"Found {len(image_files)} images to resize")
    
    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                resized_img = img.resize(size, Image.Resampling.LANCZOS)
                output_file = output_path / img_file.name
                resized_img.save(output_file)
                print(f"Resized: {img_file.name}")
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    print(f"Completed! Resized images saved to {output_folder}")

# resize_images_in_folder(
#     input_folder=r"C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\train\images",
#     output_folder=r"C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\train\images_resized",
#     size=(256, 256)
# )
# resize_images_in_folder(
#     input_folder=r"C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\train\masks",
#     output_folder=r"C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\train\masks_resized",
#     size=(256, 256)
# )

# resize_images_in_folder(
#     input_folder=r"C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\test",
#     output_folder=r"C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\test_resized",
#     size=(256, 256)
# )

# resize_images_in_folder(
#     input_folder=r"C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\val",
#     output_folder=r"C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\val_resized",
#     size=(256, 256)
# )
# dir = ...
# #clean train folder
# for file in os.listdir(dir):
#     if 'mask' in file:
#         print(f'moving file: {file} to masks')
#         shutil.move(os.path.join(dir,file),...)
#     elif 'sat' in file:
#         print(f'moving file: {file} to images')
#         shutil.move(os.path.join(dir,file), ...)
#     else:
#         pass

#transform data for use
# IMAGES_DIR = r'C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\train\images'
# MASKS_DIR = r'C:\Users\10295498\OneDrive - BD\Desktop\deepglobe_cv\deepglobe_cv\data\data_files\train\masks'
images = [] 

# print('transforming images to .pkl')
# for filename in os.listdir(IMAGES_DIR):
#     if filename.lower().endswith(".jpg"):
#         path = os.path.join(IMAGES_DIR, filename) 
#         img = Image.open(path).convert("RGB") 
#         img = np.array(img) 
#         images.append(img) 
#         with open("images.pkl", "wb") as f:
#             pickle.dump(images, f)
# print('='*60)
# print('complete!')
# print('')
# print('transforming masks to .pkl')
# for filename in os.listdir(MASKS_DIR):
#     if filename.lower().endswith(".jpg"):
#         path = os.path.join(MASKS_DIR, filename) 
#         img = Image.open(path).convert("RGB") 
#         img = np.array(img) 
#         images.append(img) 
#         with open("masks.pkl", "wb") as f:
#             pickle.dump(images, f)
# print('='*60)
# print('complete!')

COLOR_MAP = { (0, 255, 255): 0, # Urban 
             (255, 255, 0): 1, # Agriculture 
             (255, 0, 255): 2, # Rangeland 
             (0, 255, 0): 3, # Forest 
             (0, 0, 255): 4, # Water 
             (255, 255, 255): 5,# Barren 
             (0, 0, 0): 6 # Unknown 
             }

class DeepGlobeDataset:
    def __init__(
            self,
            images_path: str,
            masks_path: str,
            num_classes: int = 7,
            augmentation: Optional[callable] = None,
            class_weights: Optional[dict] = None,
    ):
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.class_weights = class_weights
        self.images = images_path
        self.masks = masks_path
        assert len(self.images) == len(self.masks), "Images and masks must have same length"

        print(f"loaded {len(self.iamges)} samples")

        def __len__(self):
            return len(self.images)
        
        import numpy as np

    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        # mask is H x W x 3 (RGB)
        h, w, _ = mask.shape
        class_mask = np.zeros((h, w), dtype=np.int32)

        for rgb, cls in COLOR_MAP.items():
            matches = np.all(mask == rgb, axis=-1)
            class_mask[matches] = cls

        # Clip and one-hot encode 
        class_mask = np.clip(class_mask, 0, self.num_classes - 1)
        one_hot = tf.keras.utils.to_categorical(class_mask, self.num_classes)
        return one_hot.astype(np.float32)
    
    def split_dataset(self, val_split: float = 0.2, random_state: int = 42):
        indices = np.arange(len(self.images))
        train_idx, val_idx = train_test_split(
            indices,
            test_size = val_split,
            random_state=random_state
        )

        #create train dataset
        train_images = [self.images[i] for i in train_idx]
        train_masks = [self.masks[i] for i in train_idx]
        train_dataset = DeepGlobeDataset.__create_from_lists(
            train_images, train_masks, self.num_classes, self.augmentation, self.class_weights
        )

        #validation dataset (no aug)
        val_images = [self.images[i] for i in val_idx]
        val_masks = [self.images[i] for i in val_idx]
        val_dataset = DeepGlobeDataset.__create_from_lists(
            val_images, val_masks, self.num_classes, None, self.class_weights
        )

        return train_dataset, val_dataset
    
    @staticmethod
    def __create_from_lists(
        images: list[np.ndarray],
        masks: list[np.ndarray],
        num_classes: int,
        augmentation: Optional[callable] = None,
        class_weights: Optional[dict] = None,
        ):
        """"helper method to create dataset from lists"""
        dataset = object.__new__(DeepGlobeDataset)
        dataset.images = images
        dataset.masks = masks
        dataset.num_classes = num_classes
        dataset.augmentation = augmentation
        dataset.class_weights = class_weights
        
        return dataset
    


    def get_tf_dataset(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """tensorflow dataset with preprocessing and augmentation"""
        images = np.array([img[..., np.newaxis] if img.ndim == 2 else img for img in self.images], dtype=np.float32) #... loops through every iamge in imageS
        masks = np.array([self.preprocess_mask(mask) for mask in self.masks], dtype=np.float32)

        #normalize step
        images = images / (images.max() + 1e-8)

        #dataset creation
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.images), reshuffle_each_iteration=True)

        if self.augmentation:
            dataset = dataset.map(self.augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    
    def get_class_distribution(self) -> dict:
        """class distributio nacross all masks"""
        class_counts = {i:0 for i in range (self.num_classes)}

        for mask in self.masks:
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[int(cls)] += count
        
        return class_counts