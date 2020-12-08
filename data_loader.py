import numpy as np
import cv2
import torch
import torch.utils.data as data

def load_data(dataframe, batch=16, file_type='jpg'):
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    if file_type == 'jpg':
        X = [cv2.imread(dataframe.iloc[i,0], 1) for i in range(batch)]
        Y = np.array([cv2.imread(dataframe.iloc[i,1], 1) for i in range(batch)])
    else:
        X = [np.load(dataframe.iloc[i,0]) for i in range(batch)]
        Y = np.array([np.load(dataframe.iloc[i,1]) for i in range(batch)])
    h, w, c = Y[0].shape
    X = np.array([cv2.resize(x, (w,h)) for x in X])
    X = np.transpose(X, (0,3,1,2))
    Y = np.transpose(Y, (0,3,1,2))

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    return X, Y