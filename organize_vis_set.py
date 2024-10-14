import os
import torch
import pathlib
import argparse
from utils import UNetModel
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as T
from os.path import join


def detect_tissue(model, img, threshold):
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
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
        tissue_mask = tissue_probs > threshold

    return tissue_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--dst', type=str, default="./splits", help='Path to the output directory')
    parser.add_argument('--model_path', type=str, default="./utils/fcn-tissue_mask.pth", help='Path to the tissue mask model')
    parser.add_argument('--tissue_threshold', type=float, default=.5, help='threshold to decide tissue or not')
    parser.add_argument('--drop_threshold', type=float, default=.005, help='threshold to drop the patch')
    parser.add_argument('--gpu', type=str, default='2', help='GPU index')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = UNetModel(num_input_channels=3, num_output_channels=2, encoder='resnet50', decoder_block=[3])
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    df = pd.DataFrame(columns=['slide_id', 'patch_id'])

    wsi_idx = [f for f in os.listdir(join(args.src, 'Images')) if os.path.isdir(join(args.src, 'Images', f))]
    for wsi in tqdm(wsi_idx, desc="WSIs", position=0):
        patch_list = [f for f in os.listdir(join(args.src, 'Images', wsi)) if f.endswith('.jpeg')]
        for patch in tqdm(patch_list, desc="Patches", position=1, leave=False):
            patch_name = pathlib.Path(patch).stem
            img_path = join(args.src, 'Images', wsi, patch)
            img = Image.open(img_path).convert('RGB')
            tissue_mask = detect_tissue(model, img, args.tissue_threshold)
            tissue_ratio = np.average(tissue_mask)
            if tissue_ratio > args.drop_threshold:
                row = {'slide_id': wsi, 'patch_id': patch_name}
                df = df._append(row, ignore_index=True)

        df.to_csv(join(args.dst, 'test_visualize.csv'), index=False)
