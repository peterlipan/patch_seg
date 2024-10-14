import os
import cv2
import torch
import shutil
import random
import pathlib
import argparse
import numpy as np
from utils import UNetModel
from PIL import Image
from tqdm import tqdm
from os.path import join
import torch.nn.functional as F
import torchvision.transforms as T

# tissue mask model path: https://tiatoolbox.dcs.warwick.ac.uk/models/seg/fcn-tissue_mask.pth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter patches by tumor area')
    parser.add_argument('--src', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--dst', type=str, default="/home/r20user17/Documents/tiles_512_10x_Task1_Resampled", help='Path to the output directory')
    parser.add_argument('--model_path', type=str, default="/home/r20user17/patch_seg/utils/fcn-tissue_mask.pth", help='Path to the tissue mask model')
    parser.add_argument('--size', type=int, default=512, help='Size of the images')
    parser.add_argument('--tissue_threshold', type=float, default=.5, help='threshold to decide tissue or not')
    parser.add_argument('--drop_threshold', type=float, default=.001, help='threshold of the wrong label ratio')
    parser.add_argument('--gpu', type=str, default='1', help='GPU index')
    parser.add_argument('--background', type=int, default=255, help='Intensity of the target class. (255=White)')
    args = parser.parse_args()
    os.makedirs(os.path.join(args.dst, 'Images'), exist_ok=True)
    os.makedirs(os.path.join(args.dst, 'Labels'), exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = UNetModel(num_input_channels=3, num_output_channels=2, encoder='resnet50', decoder_block=[3])
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
    ])

    # >1 values just to ensure the class can be selected
    # not mean over sample
    class2keep = {'Tumor': .6, 'Normal cartilage': 1.1, 'Bone': 1.1, 'Marrow': 1.1, 'Soft tissue': .3, 'Background': 1.1}
    class2grey = {'Tumor': 59, 'Normal cartilage': 178, 'Bone': 33, 'Marrow': 145, 'Soft tissue': 75, 'Background': 255}

    src_wsi_idx = [f for f in os.listdir(join(args.src, 'Images')) if os.path.isdir(join(args.src, 'Images', f))]
    dst_wsi_idx = [f for f in os.listdir(join(args.dst, 'Images')) if os.path.isdir(join(args.dst, 'Images', f))]
    target_wsi_idx = [f for f in src_wsi_idx]

    for wsi in tqdm(target_wsi_idx, desc="WSIs", position=0):
    # for wsi in tqdm(src_wsi_idx, desc="WSIs", position=0):
        os.makedirs(os.path.join(args.dst, 'Images', wsi), exist_ok=True)
        os.makedirs(os.path.join(args.dst, 'Labels', wsi), exist_ok=True)
        
        patch_count = 0  # Initialize patch count for each WSI

        for patch in tqdm([f for f in os.listdir(join(args.src, 'Labels', wsi)) if f.endswith('.png')], desc="Patches", position=1, leave=False):
            patch_name = pathlib.Path(patch).stem
            img_path = join(args.src, 'Images', wsi, f"{patch_name}.jpeg")
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).cuda()

            with torch.no_grad():
                probs = model(img)
                probs = F.softmax(probs, dim=1)
                probs = F.interpolate(
                    probs,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
                probs = probs.permute(0, 2, 3, 1) # to NHWC
                tissue_probs = probs[0, :, :, 1].cpu().numpy()
                tissue_mask = tissue_probs > args.tissue_threshold
            
            label = cv2.imread(join(args.src, 'Labels', wsi, patch), cv2.IMREAD_GRAYSCALE)
            
            label_bg = label == args.background
            # area of the regions that are labeldd as background but contain tissues
            label_bg_but_tissue = np.average(np.logical_and(label_bg, tissue_mask))

            # if the area is too large, drop this sample as we do not know the label of the tissue that is labeled as bg
            if label_bg_but_tissue > args.drop_threshold:
                continue

            # regions that are not labeled as background but do not contain tissues
            # seems not reasonable as the people who labeled the data know what they are doing
            # label_no_bg_but_no_tissue = np.logical_and(~label_bg, ~tissue_mask)
            # label[label_no_bg_but_no_tissue] = args.background

            label_colors = np.unique(label)

            ratios = []
            for c, v in class2grey.items():
                if v in label_colors:
                    ratios.append(class2keep[c])
            ratio = np.max(ratios) if len(ratios) > 0 else 0
            this = random.random() < ratio
            if this:
                shutil.copy(join(args.src, 'Images', wsi, f"{patch_name}.jpeg"), join(args.dst, 'Images', wsi, f"{patch_name}.jpeg"))
                cv2.imwrite(join(args.dst, 'Labels', wsi, patch), label)
