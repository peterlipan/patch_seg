import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing


ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax' 
DEVICE = 'cuda'
BATCH_SIZE = 48
EPOCHS = 20

root = '/home/wqzhao/Documents/Max/li/tiles_512_20x_MultiClass_WhiteBackGround'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
it_classes = ['Soft tissue', 'Tumor', 'Bone', 'Marrow', 'Normal cartilage']
ct_classes = ['Dedifferentiated', 'G1', 'G2', 'G3']


if __name__ == '__main__':
    class_color_csv = pd.read_csv('./class_color_idx.csv')
    classes = it_classes if args.task == 'IdentifyTumor' else ct_classes
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(classes), 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = PatchDataset(
        root, 
        None, "train_multiclass.csv", 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_color_csv=class_color_csv,
    )

    valid_dataset = PatchDataset(
        root, 
        None, "valid_multiclass.csv",
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_color_csv=class_color_csv,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    loss = utils.losses.DiceLoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5),
        utils.metrics.Fscore(),
        utils.metrics.Accuracy(),
        utils.metrics.Precision(),
        utils.metrics.Recall(),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    max_score = 0
    for i in range(EPOCHS):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, f'unet_{ENCODER}.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')