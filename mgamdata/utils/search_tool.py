import os
from typing_extensions import deprecated



# NOTE 输入的所有gt文件夹是有优先级顺序的，只会返回最先找到的gt路径
@deprecated("Search Tool is out of date.")
def search_mha_file(source_folders:list[str]|str, seriesUID:str, target_type:str|None):
    if isinstance(source_folders, str):
        source_folders = [source_folders]
    assert target_type in ['image', 'label', None]
    for source_folder in source_folders:
        for roots, dirs, files in os.walk(source_folder):
            for file in files:
                if file.rstrip('.mha')==seriesUID:
                    if target_type is None or target_type in os.path.split(roots):
                        return os.path.join(roots, file)
    else:
        print(f"Can't find {target_type} file, UID {seriesUID}, type: {target_type}.")

@deprecated("Search Tool is out of date.")
def find_sample_pair(image_folder:str, label_folder:str):
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)
    sample_pairs = []
    for label_file in label_files:
        if label_file.endswith('.mha') and label_file in image_files:
            image_file = label_file.replace('label', 'image')
            sample_pairs.append((os.path.join(image_folder, image_file),
                                 os.path.join(label_folder, label_file)))
    return sample_pairs
