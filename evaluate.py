import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing


ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 96
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

if __name__ == '__main__':
    root = '/home/r20user17/Documents/tiles_1024_10x_blackisTumor'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    best_model = torch.load('unet_resnet18.pth')
    valid_dataset = PatchDataset(
        root, 
        None, "valid.csv",
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    loss = utils.losses.DiceLoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5),
        utils.metrics.Fscore(),
        utils.metrics.Accuracy(),
        utils.metrics.Precision(),
        utils.metrics.Recall(),
    ]

    test_epoch = utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device='cuda',
    )

    logs = test_epoch.run(valid_dataloader)

    print(logs)
