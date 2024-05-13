import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing



ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 96
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
DEVICE = 'cuda'


# helper function for data visualization
def visualize(images, idx):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(f'vis/visualize_{idx}.png')


if __name__ == '__main__':
    root = '/home/r20user17/Documents/tiles_1024_10x_blackisTumor'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    best_model = torch.load('unet_resnet34.pth')
    valid_dataset = PatchDataset(
        root, 
        None, "valid.csv",
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset_vis = PatchDataset(
        root,
        None, "valid.csv", 
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

    for i in range(100):
        n = np.random.choice(len(valid_dataset))
        
        image_vis = valid_dataset_vis[n][0].astype('uint8')
        image, gt_mask = valid_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        visualize(
            images={'image': image_vis, 'ground_truth_mask': gt_mask, 'predicted_mask': pr_mask},
            idx = i
        )
