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
    train_shape = img.shape
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img[0, train_shape[1]//5], cmap='gray')
    ax[1].imshow(mask[0, train_shape[1]//5], cmap='gray')
    os.makedirs('./InjectVisualize', exist_ok=True)
    fig.savefig(f'./InjectVisualize/visualize_{time()}.png')



