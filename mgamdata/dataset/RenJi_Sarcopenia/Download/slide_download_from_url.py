import pandas as pd
import requests
import numpy as np
import os
import shutil

def slide_download_from_url(csv_path, save_dir, slide_pool_path=None):
    data = pd.read_csv(csv_path)
    data_len = len(data)
    
    for i in np.arange(data_len):
        num, ids, file_name, link, _, _ = data.iloc[i]
        #print(num, ids, link)
        slide_dir = os.path.join(save_dir, num)
        #if os.path.exists(slide_dir):
        #    continue
    
        if slide_pool_path:
            if os.path.exists(os.path.join(slide_pool_path, num)):
                print("exist pool: " + num)
                shutil.move(os.path.join(slide_pool_path, num), save_dir)
                continue
            
        img_response = requests.get(link)
        file_name = file_name.replace(" ", "")

        save_path = os.path.join(slide_dir, file_name)  
        print(save_path)
        os.makedirs(slide_dir, exist_ok=True)
        
        with open(save_path,'wb') as f:
            print(len(img_response.content))
            f.write(img_response.content)

def slide_download_from_url_(csv_path, save_dir, slide_pool_path=None):
    data = pd.read_csv(csv_path)
    data_len = len(data)
    print(data_len)
    for i in np.arange(data_len):
        ss_data_i = data.iloc[i]
        num = ss_data_i['序列号']
        link = ss_data_i['文件内网地址']
        
        file_name = ss_data_i['文件名']
        print(file_name)
        #num, ids, file_name, link, _, _ = data.iloc[i]
    
        slide_dir = os.path.join(save_dir, num)
        #if os.path.exists(slide_dir):
        #    continue
    
        if slide_pool_path:
            if os.path.exists(os.path.join(slide_pool_path, num)):
                print("exist pool: " + num)
                shutil.move(os.path.join(slide_pool_path, num), save_dir)
                continue
            
        img_response = requests.get(link)
        file_name = file_name.replace(" ", "")

        save_path = os.path.join(slide_dir, file_name)  
        print(save_path)
        os.makedirs(slide_dir, exist_ok=True)
        
        with open(save_path,'wb') as f:
            print(len(img_response.content))
            f.write(img_response.content)            
      
    
if __name__ == '__main__':
    csv_path = '/fileser51/zhangwh.lw/workspace/Data-Download/CSV_files/ultrasound/sid_url.csv'
    save_dir = "/fileser51/zhangwh.lw/workspace/Data-Download/ultrasound"
    slide_pool_path = None
    slide_download_from_url_(csv_path, save_dir, slide_pool_path)
    print("download finished !")