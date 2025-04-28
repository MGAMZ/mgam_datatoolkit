import pandas as pd
from tqdm import tqdm
import requests
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
import os


if __name__ == '__main__':

    # data = pd.read_csv("./文件内网地址信息-导出结果.csv")

    save_dir = "/fileser51/DataPathology/renji_sarcopenia/"
    data = pd.concat(pd.read_excel("./文件内网地址信息-导出结果.xlsx", sheet_name=None, skiprows=0))
    for i in tqdm(range(len(data))):
        num, ids, file_name, link, _,er = data.iloc[i]
        print (num, ids, link)
        img_response = requests.get(link)
        img_name = link.split("/")[-1].split("?")[0]
        if os.path.exists(save_dir + num +"/" +str(img_name)):
            continue
        else:
            # if 'Thumbs' in file_name:
            os.makedirs(save_dir + num +"/", )
            save_path = save_dir + num +"/" +str(img_name)
            filename = str(save_path)
            print(filename)
            with open(filename,'wb') as f:
                f.write(img_response.content)