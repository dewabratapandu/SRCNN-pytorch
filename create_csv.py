import pandas as pd
import os
import re
import numpy as np
import cv2

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

lr_path = "DIV2K_train_LR_x8"
hr_path = "DIV2K_train_HR"
lr = [os.path.join(lr_path, s) for s in sorted(os.listdir(lr_path), key=numericalSort)]
hr = [os.path.join(hr_path, s) for s in sorted(os.listdir(hr_path), key=numericalSort)]

df = pd.DataFrame(columns=['lr', 'hr'])
for l, h in zip(lr, hr):
    df = df.append({'lr': l, 'hr': h}, ignore_index=True)

    im_l = cv2.imread(l, 1)
    im_l = cv2.resize(im_l, (480, 270))
    cv2.imwrite(l, im_l)
    im_h = cv2.imread(h, 1)
    im_h = cv2.resize(im_h, (1920, 1080))
    cv2.imwrite(h, im_h)

print(df.iloc[:5, :])
df.to_csv('train.csv', index=False)