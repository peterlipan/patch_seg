import os
import pathlib
import pandas as pd
from tqdm import tqdm


img_dir = '/home/wqzhao/Documents/Max/li/tiles_512_20x_MultiClass_WhiteBackGround/Images'
msk_dir = '/home/wqzhao/Documents/Max/li/tiles_512_20x_MultiClass_WhiteBackGround/Labels'

img_wsi = os.listdir(img_dir)
msk_wsi = os.listdir(msk_dir)
# assert set(img_wsi) == set(msk_wsi)

exist_test_df = pd.read_csv('valid.csv')
exist_train_df = pd.read_csv('train.csv')
test_slide_idx = exist_test_df['slide_id'].unique()
train_slide_idx = exist_train_df['slide_id'].unique()

exclud_id = []

# generate the new train.csv
train_df = pd.DataFrame(columns=['slide_id', 'patch_id'])
for id in tqdm(train_slide_idx,  desc="Training set", position=0):
    if id in exclud_id:
        continue
    subpath = os.path.join(img_dir, id)
    if not os.path.exists(subpath):
        continue
    for patch_id in tqdm(os.listdir(subpath), desc="Patches", position=1, leave=False):
        if patch_id.endswith('.jpeg') and patch_id.startswith('CHS'):
            patch_name = pathlib.Path(patch_id).stem
            msk_path = os.path.join(msk_dir, id, f'{patch_name}.png')
            if not os.path.exists(msk_path):
                continue
            train_df = train_df._append({'slide_id': id, 'patch_id': patch_name}, ignore_index=True)
train_df.to_csv('train_multiclass.csv', index=False)

# generate the new valid.csv
valid_df = pd.DataFrame(columns=['slide_id', 'patch_id'])
for id in tqdm(test_slide_idx, desc="Validation set"):
    if id in exclud_id:
        continue
    subpath = os.path.join(img_dir, id)
    if not os.path.exists(subpath):
        continue
    for patch_id in tqdm(os.listdir(subpath), desc="Patches", position=1, leave=False):
        if patch_id.endswith('.jpeg') and patch_id.startswith('CHS'):
            patch_name = pathlib.Path(patch_id).stem
            msk_path = os.path.join(msk_dir, id, f'{patch_name}.png')
            if not os.path.exists(msk_path):
                continue
            valid_df = valid_df._append({'slide_id': id, 'patch_id': patch_name}, ignore_index=True)
valid_df.to_csv('valid_multiclass.csv', index=False)

