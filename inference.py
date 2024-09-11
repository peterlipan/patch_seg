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


encoder = 'resnet18'
task = 'IdentifyTumor'
root = '/home/r20user17/Documents/tiles_512_10x_FiveClass_IdentifyTumor'
dst = '/home/r20user17/Documents/Predictions_IdentifyTumor_x10_512'
bs = 24
threshold = .5
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    it_classes = ['Background', 'Soft tissue', 'Tumor', 'Bone', 'Marrow', 'Normal cartilage']
    ct_classes = ['Background', 'Dedifferentiated', 'G1', 'G2', 'G3']
    target_cls = it_classes if task == 'IdentifyTumor' else ct_classes
    class_color_csv = pd.read_csv('./class_color_idx.csv')

    valid_csv = pd.read_csv(f"./valid_{task}.csv")
    all_slide_idx = valid_csv['slide_id'].unique()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    print(f"Inferencing {task}_{encoder}_{decoder}.pth on {len(all_slide_idx)} slides...")
    best_model = torch.load(f'{task}_{encoder}_{decoder}.pth').cuda()

    for tar in target_cls[1:]:
        os.makedirs(os.path.join(dst, tar), exist_ok=True) 

    for specific_id in tqdm(all_slide_idx, position=0, desc="Slide"):
        valid_dataset = PatchDataset(
            root, 
            None, f"./valid_{task}.csv",
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            specific_slide=specific_id,
            inference=True,
            class_color_csv=class_color_csv,
            task=args.task,
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
        # slide_path = os.path.join(dst, specific_id)
        # os.makedirs(slide_path, exist_ok=True)

        with torch.no_grad():
            best_model.eval()
            for image, patch_id in tqdm(valid_dataloader, position=1, desc="Patch", leave=False):
                image = image.cuda(non_blocking=True)

                pr_mask = best_model.predict(image)
                pr_mask = np.argmax(pr_mask.squeeze().cpu().numpy(), axis=0)
                # save the mask
                mask = preds.cpu().numpy().squeeze()
                mask = mask.astype('uint8')
                mask = cv2.resize(mask, (512, 512)).astype('uint8')
                save_name = f'Labels_{patch_id[0]}.tif'
                save_name = save_name.replace('w=1024,h=1024', 'w=512,h=512')
                cv2.imwrite(os.path.join(dst, save_name), mask)               
            