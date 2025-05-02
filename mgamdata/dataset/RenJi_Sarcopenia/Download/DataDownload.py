import os
import requests
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    data = pd.read_csv("./URL_Result_8009.csv")
    save_dir = "/fileser51/zhangyiqin.sx/Sarcopenia_Data/batch5_8009/"
    # data = pd.concat(pd.read_csv("./URL_Result_8009.csv"))
    
    for i in tqdm(range(len(data))):
        num, ids, file_name, link, _,er = data.iloc[i]
        print (num, ids, link)
        img_response = requests.get(link)
        img_name = link.split("/")[-1].split("?")[0]
        if os.path.exists(save_dir + num +"/" +str(img_name)):
            continue
        else:
    #    if 'Thumbs' in file_name:
            os.makedirs(save_dir + num +"/", exist_ok=True)
            save_path = save_dir + num +"/" +str(img_name)
            filename = str(save_path)
            print(filename)
            with open(filename,'wb') as f:
                f.write(img_response.content)