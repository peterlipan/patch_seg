import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing


ENCODER = 'efficientnet-b1'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 24
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    root = '/home/r20user17/Documents/tiles_512_20X_WhiteIsTumor_256overlap'
    dst = '/home/r20user17/Documents/predicted_masks_512'
    threshold = .5
    valid_csv = pd.read_csv("./valid_512.csv")
    all_slide_idx = valid_csv['slide_id'].unique()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    print(f"Inferencing unet_{ENCODER}.pth on {len(all_slide_idx)} slides...")
    best_model = torch.load(f'unet_{ENCODER}.pth').cuda()
    os.makedirs(dst, exist_ok=True)

    for specific_id in tqdm(all_slide_idx, position=0, desc="Slide"):
        valid_dataset = PatchDataset(
            root, 
            None, "valid_512.csv",
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            specific_slide=specific_id,
            inference=True
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
        # slide_path = os.path.join(dst, specific_id)
        # os.makedirs(slide_path, exist_ok=True)

        with torch.no_grad():
            for image, patch_id in tqdm(valid_dataloader, position=1, desc="Patch", leave=False):
                image = image.cuda(non_blocking=True)

                preds = best_model(image)
                preds = (preds > threshold).float()
                # save the mask
                mask = preds.cpu().numpy().squeeze()
                mask = mask.astype('uint8')
                mask = cv2.resize(mask, (512, 512)).astype('uint8')
                save_name = f'Labels_{patch_id[0]}.tif'
                save_name = save_name.replace('w=1024,h=1024', 'w=512,h=512')
                cv2.imwrite(os.path.join(dst, save_name), mask)               
            