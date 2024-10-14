import os
import pathlib
import pandas as pd
from tqdm import tqdm


img_dir = '/home/r20user17/Documents/tiles_512_10x_Task1_Resampled/Images'
msk_dir = '/home/r20user17/Documents/tiles_512_10x_Task1_Resampled/Labels'

img_wsi = os.listdir(img_dir)
msk_wsi = os.listdir(msk_dir)
# assert set(img_wsi) == set(msk_wsi)

train_slide_idx = pd.read_csv('./splits/train_set.csv')['slide_id'].unique()
test_slide_idx = pd.read_csv('./splits/test_set.csv')['slide_id'].unique()

# generate the new train.csv
train_df = pd.DataFrame(columns=['slide_id', 'patch_id'])
for id in tqdm(train_slide_idx,  desc="Training set", position=0):
    subpath = os.path.join(img_dir, id)
    if not os.path.exists(subpath):
        continue
    for patch_id in tqdm(os.listdir(subpath), desc="Patches", position=1, leave=False):
        if patch_id.endswith('.jpeg'):
            patch_name = pathlib.Path(patch_id).stem
            msk_path = os.path.join(msk_dir, id, f'{patch_name}.png')
            if not os.path.exists(msk_path):
                continue
            train_df = train_df._append({'slide_id': id, 'patch_id': patch_name}, ignore_index=True)
train_df.to_csv('./splits/train_IdentifyTumor.csv', index=False)

# generate the new valid.csv
test_df = pd.DataFrame(columns=['slide_id', 'patch_id'])
for id in tqdm(test_slide_idx, desc="Validation set"):
    subpath = os.path.join(img_dir, id)
    if not os.path.exists(subpath):
        continue
    for patch_id in tqdm(os.listdir(subpath), desc="Patches", position=1, leave=False):
        if patch_id.endswith('.jpeg'):
            patch_name = pathlib.Path(patch_id).stem
            msk_path = os.path.join(msk_dir, id, f'{patch_name}.png')
            if not os.path.exists(msk_path):
                continue
            test_df = test_df._append({'slide_id': id, 'patch_id': patch_name}, ignore_index=True)
test_df.to_csv('./splits/test_IdentifyTumor.csv', index=False)

