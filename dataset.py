import os
import cv2
import torch
import pathlib
import numpy as np
import pandas as pd
import albumentations as albu
from utils import RandomImageMaskShuffle
from torch.utils.data.dataset import Dataset


def get_training_augmentation():
    train_transform = [
        albu.Resize(height=512, width=512),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
                albu.MedianBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),

        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        albu.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        
        # albu.GridDropout(ratio=0.5, unit_size_min=64, unit_size_max=128, random_offset=True, fill_value=255, mask_fill_value=0, p=0.5),

    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(height=512, width=512),
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
            specific_slide=None,
            inference=False,
            class_color_csv=None,
            task = 'IdentifyTumor'
    ):
        if csv_path is None:
            df = pd.DataFrame(columns=['slide_id', 'patch_id'])
            for item in slide_ids:
                subpath = os.path.join(data_root, 'Images', item)
                for patch_id in os.listdir(subpath):
                    if patch_id.endswith('.jpeg') and patch_id.startswith('CHS'):
                        df = df._append({'slide_id': item, 'patch_id': pathlib.Path(patch_id).stem}, ignore_index=True)
            self.df = df
        else:
            self.df = pd.read_csv(csv_path)
        if specific_slide is not None:
            self.df = self.df[self.df['slide_id'] == specific_slide]
        
        self.data_root = data_root
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.inference = inference
        self.random_shuffle = RandomImageMaskShuffle(size=512, ratio_up=0.8, ratio_low=0.3, p=0.5)

        it_classes = ['Background', 'Soft tissue', 'Tumor', 'Bone', 'Marrow', 'Normal cartilage']
        ct_classes = ['Background', 'Dedifferentiated', 'G1', 'G2', 'G3']
        class2gray = dict(zip(class_color_csv['class'], class_color_csv['gray']))
        class2rgb = dict(zip(class_color_csv['class'], zip(class_color_csv['r'], class_color_csv['g'], class_color_csv['b'] )))

        if task == 'IdentifyTumor':
            self.class_values = [class2gray[c] for c in it_classes]
            self.class_rgbs = [class2rgb[c] for c in it_classes]
        elif task == 'ClassifyTumor':
            self.class_values = [class2gray[c] for c in ct_classes]
            self.class_rgbs = [class2rgb[c] for c in ct_classes]
    
    @staticmethod
    def rgb2gray(r, g, b):
        return int(0.299 * r + 0.587 * g + 0.114 * b)
    
        
    def __getitem__(self, i):
        try:
            # read data
            slide_id = self.df['slide_id'].values[i]
            patch_id = self.df['patch_id'].values[i]
            image = cv2.imread(os.path.join(self.data_root, 'Images', slide_id, patch_id+'.jpeg'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.inference:
                if self.augmentation:
                    image = self.augmentation(image=image)['image']            
            
                # apply preprocessing
                if self.preprocessing:
                    image = self.preprocessing(image=image)['image']
            
                return image, patch_id
            mask = cv2.imread(os.path.join(self.data_root, 'Labels', slide_id, patch_id+'.png'), 0)
            
            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
        
        
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = self.random_shuffle(image=sample['image'], mask=sample['mask'])
        
        
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            
            return image, mask

        except Exception as e:
            print("Error! ", e)
            print(f"patch id: {patch_id}")


    def __len__(self):
        return self.df.shape[0]
