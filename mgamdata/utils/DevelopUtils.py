import os
from time import time

import torch
import numpy as np


# function decorator for debug
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"{func.__qualname__} 执行时间: {elapsed_time:.2f} 秒")
        return result
    return wrapper


# for debug
def InjectVisualize(img, mask):
    import matplotlib.pyplot as plt
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy() # [B, D, H, W]
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy() # [B, D, H, W]
    if img.ndim == 3:
        img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img[0, img.shape[1]//2], cmap='gray')
    ax[1].imshow(mask[0, img.shape[1]//2], cmap='rainbow')
    os.makedirs('./InjectVisualize', exist_ok=True)
    fig.savefig(f'./InjectVisualize/visualize_{time()}.png')


# for debug
def InjectVisualize_2D(img1, img2):
    import matplotlib.pyplot as plt
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach().numpy() # [C, H, W]
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach().numpy() # [C, H, W]
    if img1.ndim == 3:
        img1 = img1.transpose(1, 2, 0)
    if img2.ndim == 3:
        img2 = img2.transpose(1, 2, 0)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    
    # show min, max, mean, std
    ax[0].text(0, 0, f"min: {img1.min():.2f}\nmax: {img1.max():.2f}\nmean: {img1.mean():.2f}\nstd: {img1.std():.2f}", fontsize=8, color='blue', transform=ax[0].transAxes)
    ax[1].text(0, 0, f"min: {img2.min():.2f}\nmax: {img2.max():.2f}\nmean: {img2.mean():.2f}\nstd: {img2.std():.2f}", fontsize=8, color='blue', transform=ax[1].transAxes)
    
    os.makedirs('./InjectVisualize', exist_ok=True)
    fig.savefig(f'./InjectVisualize/visualize_{time()}.png')
