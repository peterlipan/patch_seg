import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


class_color_df = pd.read_csv('./class_color_idx.csv')
classes = class_color_df['class'].values.tolist()[1:]
class_colors = class_color_df['gray'].values.tolist()[1:]
df = pd.DataFrame(columns=['slide_id'] + classes)

parse = argparse.ArgumentParser()
parse.add_argument('--src', type=str, default='path to the data')
args = parse.parse_args()

slide_idx = [f for f in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, f))]

for sid in tqdm(slide_idx, desc='slides', position=0):

    row = {'slide_id': sid}
    for c in classes:
        row[c] = 0
    label_dir = os.path.join(args.src, sid)
    label_idx = [f for f in os.listdir(label_dir) if f.endswith('.png')]

    for label in tqdm(label_idx, desc='labels', position=1, leave=False):
        label_path = os.path.join(label_dir, label)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_colors = np.unique(label)
        for i, c in enumerate(classes):
            if class_colors[i] in label_colors:
                row[c] += 1
    df = df._append(row, ignore_index=True)
    df.to_csv('./slide_class_count.csv', index=False)