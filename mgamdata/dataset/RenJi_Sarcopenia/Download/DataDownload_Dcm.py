import os
import pdb
import requests
import pandas as pd
from tqdm import tqdm


def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as out_file:
        out_file.write(response.content)


if __name__ == "__main__":
    csv_path = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Data7896/7896_URL.csv'
    save_dir = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Data7896/dcm'
    dcm_url_col_name = '文件内网地址'
    img_df = pd.read_csv(csv_path)
    
    for index, row in tqdm(img_df.iterrows(),
                           total=len(img_df),
                           dynamic_ncols=True):
        url = row[dcm_url_col_name]
        uid = str(row['序列号'])
        dir_name = os.path.join(save_dir, uid)
        os.makedirs(dir_name, exist_ok=True)
        img_name = str(row['文件编号'])
        file_path = os.path.join(save_dir, uid, img_name+'.dcm')
        if os.path.exists(file_path):
            continue
        
        try:
            download_file(url, file_path)
            tqdm.write(f"Downloading DCM: {file_path}")
        except Exception as e:
            tqdm.write(f"Error downloading DCM file {file_path}: {e}")
