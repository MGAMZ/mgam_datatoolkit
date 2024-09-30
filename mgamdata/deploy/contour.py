
import requests,json
from glob import glob
import os
import pandas as pd
import numpy as np
import torch
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
import cv2
from sklearn.cluster import KMeans
import pdb
from os import path as osp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import SimpleITK as sitk





def contours(mask):
    coordlist = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return [[]]
        
    
    for j in range(len(contours)):
        contour_temp = contours[j]
        coord_temp_list = []
        for i in range(len(contour_temp)):
            coord_temp_list.append(tuple(contour_temp[i][0]))
        coordlist.append(coord_temp_list)
    return coordlist

def findContourEachRegion(z,seglist):
    contour_list = []
    for seg_region in seglist:
        slice2d = np.uint8(seg_region[z])
        res = contours(slice2d)
        contour_list.append(res)
    return contour_list


def findCoord(mask):
    xlist,ylist = np.where(mask>0)
    coordlist = []
    for i in range(len(xlist)):
        coordlist.append((xlist[i],ylist[i]))
    return coordlist

def findEachRegion(z,seglist):
    contour_list = []
    for seg_region in seglist:
        slice2d = np.uint8(seg_region[z])
        res = findCoord(slice2d)
        contour_list.append(res)
    return contour_list


def segment_within_muscular_with_kmeans(image:np.ndarray, mask:np.ndarray):
    '''
        input:
            - image: 3D Scan, ndarray (N, H, W) (dtype=np.float32)
            - mask:  3D Scan, ndarray (N, H, W) (dtype=bool)
        output:
            - pixels:       ndarray (N, 1)
            - sub-labels:   ndarray (N, 1)
    '''
    pixels = image[mask][..., np.newaxis]   # 在image中取出mask指定的像素
    print('pixels: ',pixels.min(),pixels.max())
    kmeans = KMeans(n_clusters=2)           # 二分类，基于假设：肌肉分割中存在肌间脂肪和肌肉两者。
    kmeans.fit(pixels)                      # WARNING: 最小实现。抹除了pixel的空间关系，即kmeans的通道为单通道（HU Value）

    # 基于假设：脂肪HU比肌肉HU低
    # label定义：0:肌肉 1：脂肪。
    # 保证返回的聚类0一定是平均HU较高的那个。
    if np.mean(pixels[kmeans.labels_==0]) < np.mean(pixels[kmeans.labels_==1]):
        return pixels.squeeze(), 1-kmeans.labels_       # 如果该样本中聚类0的均值较低，则认为不符合肌肉特征，将其反转为1
    else:
        return pixels.squeeze(), kmeans.labels_         # 如果聚类0已经是较高的了，就不做处理了。


