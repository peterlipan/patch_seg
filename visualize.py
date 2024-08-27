import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split
from dataset import PatchDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing



ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 96
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
DEVICE = 'cuda'


def apply_color_map(mask, color_df):
    # Create an RGB image from a segmentation mask using colors from the DataFrame
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_id in range(color_df.shape[0]):  # Iterate over each class
        r, g, b = color_df.iloc[class_id]['r'], color_df.iloc[class_id]['g'], color_df.iloc[class_id]['b']  # Get RGB values for the class
        color_mask[mask == class_id] = (r, g, b)  # Assign the RGB color to the corresponding class
    return color_mask


# helper function for data visualization
def visualize(images, idx, color_df):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if 'mask' in name:  # If it's a mask, apply the color map
            image = apply_color_map(image, color_df)
        plt.imshow(image)

    handles = []
    for _, row in color_df.iterrows():
        color = (row['r']/255, row['g']/255, row['b']/255)  # Normalize to [0, 1]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{row["class"]}',
                                    markerfacecolor=color, markersize=10))

    plt.legend(handles=handles, title="Classes", loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.tight_layout()

    plt.savefig(f'vis/visualize_{idx}.png')
    plt.close()


if __name__ == '__main__':
    root = '/home/wqzhao/Documents/Max/li/tiles_512_20x_MultiClass_WhiteBackGround'
    class_color_csv = pd.read_csv('./class_color_idx.csv')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    best_model = torch.load('unet_resnet18.pth')
    valid_dataset = PatchDataset(
        root, 
        None, "valid_multiclass.csv",
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_color_csv=class_color_csv,
    )

    valid_dataset_vis = PatchDataset(
        root,
        None, "valid_multiclass.csv", 
        class_color_csv=class_color_csv,
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
        
        gt_mask = np.argmax(gt_mask.squeeze(), axis=0)
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = np.argmax(pr_mask.squeeze().cpu().numpy(), axis=0)
        
            
        visualize(
            images={'image': image_vis, 'ground_truth_mask': gt_mask, 'predicted_mask': pr_mask},
            idx = i, color_df=class_color_csv
        )
