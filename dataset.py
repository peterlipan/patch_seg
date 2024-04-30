import os
import cv2
import torch
import pathlib
import numpy as np
import pandas as pd
import albumentations as albu
from torch.utils.data.dataset import Dataset


def get_training_augmentation():
    train_transform = [
        albu.Resize(height=512, width=512),
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(height=512, width=512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class PatchDataset(Dataset):    
    def __init__(
            self, 
            data_root, slide_ids, csv_path=None,
            augmentation=None, 
            preprocessing=None,
    ):
        if csv_path is None:
            df = pd.DataFrame(columns=['slide_id', 'patch_id'])
            for item in slide_ids:
                subpath = os.path.join(data_root, 'Images', item)
                for patch_id in os.listdir(subpath):
                    if patch_id.endswith('.jpeg') and patch_id.startswith('CHS'):
                        df = df._append({'slide_id': item, 'patch_id': pathlib.Path(patch_id).stem}, ignore_index=True)
            self.df =df
        else:
            self.df = pd.read_csv(csv_path)
        # convert str names to class values on masks
        self.class_values = [0]
        self.data_root = data_root
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
        
    def __getitem__(self, i):
        
        # read data
        slide_id = self.df['slide_id'].values[i]
        patch_id = self.df['patch_id'].values[i]
        image = cv2.imread(os.path.join(self.data_root, 'Images', slide_id, patch_id+'.jpeg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.data_root, 'Labels', slide_id, patch_id+'.png'), 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return self.df.shape[0]
