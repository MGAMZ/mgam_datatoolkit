# coding=utf-8

import os
import pdb
import numpy as np
from os import path as osp

import SimpleITK as sitk
import pandas as pd
from skimage.exposure import equalize_hist
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from ..meta import *


IMAGE_DIR = "/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/all_img_raw/pic"
MASK_DIR = "/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/all_img_raw/mask"
BEST_KMEANS_SEGMENTE_CASE_ID_IMAGE = '1.2.840.113704.7.32.0619.2.334.3.2831164355.531.1618186171.215.3_0000.mha'
BEST_KMEANS_SEGMENTE_CASE_ID_MASK  = '1.2.840.113704.7.32.0619.2.334.3.2831164355.531.1618186171.215.3.mha'



def segment_within_muscular_with_kmeans(image:np.ndarray, mask:np.ndarray):
    # 在image中筛选来自于mask的像素
    pixels = np.take(image, np.where(mask.flatten())).T
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pixels)
    # 基于假设：脂肪HU比肌肉HU低
    # output: 0:肌肉，1:脂肪
    if np.mean(pixels[kmeans.labels_==0]) < np.mean(pixels[kmeans.labels_==1]):
        return pixels.squeeze(), 1 - kmeans.labels_
    else:
        return pixels.squeeze(), kmeans.labels_
    

def merge_image_gt(image, mask):
    image = equalize_hist(image, mask=mask)
    # image, mask: (h, w)
    # num_classes_in_mask: 7
    mask[mask==0] = np.nan
    
    plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, alpha=0.7, vmin=0, vmax=6, cmap='hsv')
    plt.axis('off')
    plt.savefig('./ImageMask.png', bbox_inches=0, dpi=512)


def draw_histogram(image, mask):
    fig, axes = plt.subplots(1, 5, figsize=(5*3, 1*3))
    
    for class_idx in CLASS_MAP.keys():
        pixels_of_the_class = image[mask==class_idx]
        hist, bins = np.histogram(pixels_of_the_class, bins=100)
        hist = (hist * 0.79*0.79/1000)
        axes[class_idx].bar(bins[:-1], hist, width=np.diff(bins), align='edge')
        # axes[class_idx].hist(pixels_of_the_class, bins=range(-200,200,20), )
        axes[class_idx].set_title(CLASS_MAP[class_idx])
        # axes[class_idx].set_ylim([0, 200])
        axes[class_idx].set_xlim([-200, 200])
        # axes[class_idx].savefig(f'./histogram_240816_{CLASS_MAP[class_idx]}.png', bbox_inches=0, dpi=600)
    fig.tight_layout()
    fig.savefig('./histogram_240816.png', bbox_inches=0, dpi=600)



if __name__ == '__main__':
    image = sitk.ReadImage(osp.join(IMAGE_DIR, BEST_KMEANS_SEGMENTE_CASE_ID_IMAGE)) # 在image路径最后添加_0000
    mask  = sitk.ReadImage(osp.join(MASK_DIR , BEST_KMEANS_SEGMENTE_CASE_ID_MASK ))
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_array  = sitk.GetArrayFromImage(mask).astype(np.float32)
    
    draw_histogram(image_array, mask_array)
    
    
    
    
    exit()
    
    
    # label override according to kmeans segmentation
    for class_id_need_kmeans in [1,2]:
        mask_need_kmeans = mask_array == class_id_need_kmeans
        _pixel_under_mask_flattened, kmeans_seg_map = segment_within_muscular_with_kmeans(image_array, mask_need_kmeans)
        # kmeans_seg_map中，0表示肌肉，1表示脂肪
        locations_fat_in_muscle = np.where(kmeans_seg_map)
        # 新加进来的两个类id，插入到末尾，是5或者6
        
        location_index = np.ravel_multi_index(np.where(mask_need_kmeans), mask_array.shape)
        selected_mask_pixels = np.take(mask_array, location_index)
        selected_mask_pixels[kmeans_seg_map==1] = class_id_need_kmeans + 4
        np.put(mask_array, location_index, selected_mask_pixels)
        # pdb.set_trace()
    
    valid_L3 = [slice for slice in range(len(mask_array)) if mask_array[slice, :, :].sum() > 0]
    selected_slice = valid_L3[len(valid_L3)//2]
    merge_image_gt(image_array[selected_slice], mask_array[selected_slice])
    
    # TASK 2:
    pixel = {value:[] for value in np.unique(mask_array)}
    contains_L3 = image_array[valid_L3]
    unique_classes = np.unique(mask_array)
    for class_idx in unique_classes:
        if class_idx == 0:
            continue
        # 获取掩码中值为value的像素点坐标
        coords = np.argwhere(mask_array == class_idx)
        for coord in coords:
            # 将像素点坐标添加到DataFrame中
            pixel[class_idx].append(image_array[coord[0], coord[1], coord[2]])

    pixel.pop(0)
    
    for key in pixel.keys():
        pixel[key] = np.sort(np.array(pixel[key]))
    
    needed_1 = np.concat([pixel[1], pixel[5]])
    needed_2 = np.concat([pixel[2] , pixel[6]])
    
    plt.figure()
    plt.title('yadaji')
    bin_edges = plt.hist(needed_1, bins=10, edgecolor='black')[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_heights = plt.hist(needed_1, bins=10, edgecolor='black')[0]
    plt.yscale('log')

    # 在柱状图顶部标注数值
    for x, y, v in zip(bin_centers, bin_heights, bin_heights):
        if v > 0:
            plt.text(float(x), float(y), '%d' % v, ha='center', va='bottom')
    for x, v in zip(bin_edges[:-1], bin_edges[1:]):
        plt.text((x+v)/2, 0, str((x+v)/2), ha='center', va='bottom')
    plt.savefig(f'./histogram_needed_1.png', bbox_inches=0, dpi=512)
    

    
    
    plt.figure()
    plt.title('gugeji')
    bin_edges = plt.hist(needed_2, bins=10, edgecolor='black')[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_heights = plt.hist(needed_2, bins=10, edgecolor='black')[0]
    plt.yscale('log')
    # 在柱状图顶部标注数值
    for x, y, v in zip(bin_centers, bin_heights, bin_heights):
        if v > 0:
            plt.text(float(x), float(y), '%d' % v, ha='center', va='bottom')
    for x, v in zip(bin_edges[:-1], bin_edges[1:]):
        plt.text((x+v)/2, 0,  str((x+v)/2), ha='center', va='bottom')
    plt.savefig(f'./histogram_needed_2.png', bbox_inches=0, dpi=512)
    
    
 


