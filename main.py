import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation


ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['tumor']
ACTIVATION = 'sigmoid' 
DEVICE = 'cuda'
BATCH_SIZE = 24
EPOCHS = 40

root = '/home/r20user17/Documents/tiles_1024_10x_blackisTumor'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



if __name__ == '__main__':
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=1, 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    slide_ids = os.listdir(os.path.join(root, 'Images'))
    patient_ids = list(set([item.split('-')[0] for item in slide_ids]))
    train_ids, valid_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_slide_ids = [item for item in slide_ids if item.split('-')[0] in train_ids]
    valid_slide_ids = [item for item in slide_ids if item.split('-')[0] in valid_ids]

    train_dataset = Dataset(
        root, 
        train_slide_ids, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = Dataset(
        root, 
        test_slide_ids, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.Accuracy(),
        smp.utils.metrics.Precision(),
        smp.utils.metrics.Recall(),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
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
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')