import os
import pdb
import requests
import pandas as pd
from tqdm import tqdm

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)


def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as out_file:
        out_file.write(response.content)


if __name__ == "__main__":
    csv_path = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch6_8016/TASK8016/image_anno_TASK_8016.csv'
    save_dir = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch6_8016/mask'
    dcm_url_col_name = '影像结果'
    img_df = pd.read_csv(csv_path)
    
    for index, row in tqdm(img_df.iterrows(),
                           total=len(img_df),
                           dynamic_ncols=True):
        url = row[dcm_url_col_name]
        uid = str(row['序列编号'])
        dir_name = os.path.join(save_dir, uid)
        os.makedirs(dir_name, exist_ok=True)
        img_name = str(row['组织类型（肌少症）'])
        file_path = os.path.join(save_dir, uid, img_name+'.mha')
        if os.path.exists(file_path):
            continue
        
        try:
            download_file(url, file_path)
            tqdm.write(f"Downloading DCM: {file_path}")
        except Exception as e:
            tqdm.write(f"Error downloading DCM file {file_path}: {e}")
