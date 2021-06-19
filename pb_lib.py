import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  

from scipy.signal import butter,filtfilt
from scipy import signal
from scipy.signal import detrend

import matplotlib.pyplot as plt

import torch
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


from torchvision import transforms


''' image dataset class'''
class ImageData(Dataset):
    def __init__(self, X, y, img_dir, transform=None, target_transform=None):

        ## provide the images and the labels (images as file names, labels as an array)
        self.imgs = X.values
        self.img_labels = y.values

        ## the folder where the images would be found specifically
        self.img_dir = img_dir

        # provide an image transform as required
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    # image reading, transforming and assigning label
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# noise removal filter
def remove_noise(data):
    N  = 3    
    Wn = 0.15
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B,A, data)
    return smooth_data
# replace outliers with an averaging filter output
def reduce_upper_outliers(df,reduce = 0.01, half_width=4):
    length = len(df.iloc[0,:])
    remove = int(length*reduce)
    for i in df.index.values: #Para cada muestra
        values = df.loc[i,:]
        sorted_values = values.sort_values(ascending = False)
        for j in range(remove): 
            idx = sorted_values.index[j]
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            for k in range(2*half_width+1):
                idx2 = idx_num + k - half_width
                if idx2 <1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.'+str(idx2)]

                count += 1
            new_val /= count # count will always be positive here
            if new_val < values[idx]: # just in case there's a few persistently high adjacent values
                df.at[i,idx] = new_val
    return df
def iterate(df, n=2):
    for i in range(n): 
        df2 = reduce_upper_outliers(df)
    return df2

# apply FFT
def fourier_transform(df):
    df_fft = np.abs(np.fft.fft(df, axis=1))
    return df_fft
# gaussian filter with sigma
def apply_filter(df):
    filt = ndimage.filters.gaussian_filter(df, sigma=10)
    return filt

def apply_normalization(df_train, df_test):
    norm_train = normalize(df_train)
    norm_test = normalize(df_test)

    return pd.DataFrame(norm_train), pd.DataFrame(norm_test)
def apply_standardization(df_train, df_test):
    scaler = StandardScaler()
    norm_train = scaler.fit_transform(df_train)
    norm_test = scaler.transform(df_test)
    
    norm_train = pd.DataFrame(norm_train)
    norm_test = pd.DataFrame(norm_test)
    return norm_train, norm_test