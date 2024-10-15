import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing


encoder = 'efficientnet-b1'
decoder = 'Unet'
task = 'IdentifyTumor'
root = '/home/r20user17/Documents/tiles_testing_set_512_x10'
dst = '/home/r20user17/Documents/Predictions_IdentifyTumor_x10_512_thre_50_b1'
threshold = 0.5
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def generate_csv(root):
    df = pd.DataFrame(columns=['slide_id', 'patch_id'])
    slides = os.listdir(os.path.join(root, 'Images'))
    for slide in slides:
        patches = [p for p in os.listdir(os.path.join(root, 'Images', slide)) if p.endswith('.jpeg')]
        for patch in patches:
            df = df._append({'slide_id': slide, 'patch_id': patch.split('.')[0]}, ignore_index=True)
    df.to_csv(f"./splits/test_visualize.csv", index=False)


if __name__ == '__main__':
    it_classes = ['Background', 'Soft tissue', 'Tumor', 'Bone', 'Marrow', 'Normal cartilage']
    ct_classes = ['Background', 'Dedifferentiated', 'G1', 'G2', 'G3']
    target_cls = it_classes if task == 'IdentifyTumor' else ct_classes
    class_color_csv = pd.read_csv('./class_color_idx.csv')

    csv_path = "./splits/test_visualize.csv"
    if not os.path.exists(csv_path):
        generate_csv(root)
    valid_csv = pd.read_csv(csv_path)
    all_slide_idx = valid_csv['slide_id'].unique()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    print(f"Inferencing {task}_{encoder}_{decoder}.pth on {len(all_slide_idx)} slides...")
    best_model = torch.load(os.path.join('./checkpoints', f'{task}_{encoder}_{decoder}.pth')).cuda()

    for tar in target_cls[1:]:
        os.makedirs(os.path.join(dst, tar), exist_ok=True) 

    for specific_id in tqdm(all_slide_idx, position=0, desc="Slide"):
        valid_dataset = PatchDataset(
            root, 
            None, csv_path,
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            specific_slide=specific_id,
            inference=True,
            class_color_csv=class_color_csv,
            task=task,
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
        
        for c in target_cls[1:]:
            os.makedirs(os.path.join(dst, c, specific_id), exist_ok=True)

        # Inside the inference loop
        with torch.no_grad():
            best_model.eval()
            for image, patch_id in tqdm(valid_dataloader, position=1, desc="Patch", leave=False):
                image = image.cuda(non_blocking=True)

                # Get predicted probabilities
                pr_mask_probs = best_model.predict(image)

                # Apply thresholding to create a binary mask
                pr_mask = np.argmax(pr_mask_probs.squeeze().cpu().numpy(), axis=0)
                pr_mask_probs = pr_mask_probs.squeeze().cpu().numpy()

                # Create a mask where probabilities < threshold are set to background class
                thresholded_mask = np.where(np.max(pr_mask_probs, axis=0) < threshold, 0, pr_mask)

                for i in range(1, len(target_cls)):
                    mask = thresholded_mask == i
                    mask = mask.astype('uint8')
                    if np.sum(mask) == 0:
                        continue
                    mask = cv2.resize(mask, (512, 512)).astype('uint8')
                    save_name = f'Labels_{patch_id[0]}.tif'
                    cv2.imwrite(os.path.join(dst, target_cls[i], specific_id, save_name), mask)

             
            