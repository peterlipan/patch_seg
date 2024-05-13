import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing


ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 96
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


if __name__ == '__main__':
    root = '/home/r20user17/Documents/tiles_1024_10x_blackisTumor'
    threshold = .5
    eps = 1e-7
    valid_csv = pd.read_csv("./valid.csv")
    all_slide_idx = valid_csv['slide_id'].unique()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    best_model = torch.load(f'unet_{ENCODER}.pth').cuda()

    iou = 0
    f1 = 0
    accuracy = 0
    precision = 0
    recall = 0

    for specific_id in tqdm(all_slide_idx):
        valid_dataset = PatchDataset(
            root, 
            None, "valid.csv",
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            specific_slide=specific_id
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        total_union = 0

        with torch.no_grad():
            for image, mask in valid_dataloader:
                image, mask = image.cuda(non_blocking=True), mask.cuda(non_blocking=True)

                preds = best_model(image)
                preds = (preds > threshold).float()

                tp = torch.sum(preds * mask).item()
                fp = torch.sum(preds).item() - tp
                fn = torch.sum(mask).item() - tp
                tn = torch.sum((1 - preds) * (1 - mask)).item()
                union = torch.sum(preds).item() + torch.sum(mask).item() - tp + eps

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                total_union += union
            
            iou += total_tp / total_union
            f1 += 2 * total_tp / (2 * total_tp + total_fp + total_fn)
            accuracy += (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
            precision += total_tp / (total_tp + total_fp)
            recall += total_tp / (total_tp + total_fn)
    
    iou /= len(all_slide_idx)
    f1 /= len(all_slide_idx)
    accuracy /= len(all_slide_idx)
    precision /= len(all_slide_idx)
    recall /= len(all_slide_idx)
    print(f"IOU: {iou}, F1: {f1}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
            