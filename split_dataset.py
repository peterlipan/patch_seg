import os
import pathlib
import pandas as pd


img_dir = '/home/r20user17/Documents/tiles_512_20X_WhiteIsTumor_256overlap/Images'
msk_dir = '/home/r20user17/Documents/tiles_512_20X_WhiteIsTumor_256overlap/Labels'

img_wsi = os.listdir(img_dir)
msk_wsi = os.listdir(msk_dir)
assert set(img_wsi) == set(msk_wsi)

exist_test_df = pd.read_csv('valid.csv')
exist_train_df = pd.read_csv('train.csv')
test_slide_idx = exist_test_df['slide_id'].unique()
train_slide_idx = exist_train_df['slide_id'].unique()

# generate the new train.csv
train_df = pd.DataFrame(columns=['slide_id', 'patch_id'])
for id in train_slide_idx:
    subpath = os.path.join(img_dir, id)
    for patch_id in os.listdir(subpath):
        if patch_id.endswith('.jpeg') and patch_id.startswith('CHS'):
            patch_name = pathlib.Path(patch_id).stem
            msk_path = os.path.join(msk_dir, id, f'{patch_name}.png')
            assert os.path.exists(msk_path), f"Missing mask for {patch_name}"
            train_df = train_df.append({'slide_id': id, 'patch_id': patch_name}, ignore_index=True)
train_df.to_csv('train_512.csv', index=False)

# generate the new valid.csv
valid_df = pd.DataFrame(columns=['slide_id', 'patch_id'])
for id in test_slide_idx:
    subpath = os.path.join(img_dir, id)
    for patch_id in os.listdir(subpath):
        if patch_id.endswith('.jpeg') and patch_id.startswith('CHS'):
            patch_name = pathlib.Path(patch_id).stem
            msk_path = os.path.join(msk_dir, id, f'{patch_name}.png')
            assert os.path.exists(msk_path), f"Missing mask for {patch_name}"
            valid_df = valid_df.append({'slide_id': id, 'patch_id': patch_name}, ignore_index=True)
valid_df.to_csv('valid_512.csv', index=False)

