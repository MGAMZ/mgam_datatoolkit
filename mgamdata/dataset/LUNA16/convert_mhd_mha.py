import os
from ..base_convert import StandardFileFormatter

class LUNA16_formatter(StandardFileFormatter):
    @staticmethod
    def _series_id(image_path: str, label_path: str) -> str:
        ...
    
    def tasks(self) -> list:
        labels_folder = os.path.join(self.args.data_root, 'seg-lungs-LUNA16')
        labels_list = [file
                       for file in os.listdir(labels_folder) 
                       if file.endswith('.mhd')]
        subset_image_folder_list = ["subset{}".format(i) for i in range(10)]
        images_list = [os.path.join(self.args.data_root, subset_image_folder, file)
                       for subset_image_folder in subset_image_folder_list
                       for file in os.listdir(os.path.join(self.args.data_root, subset_image_folder))
                       if file.endswith('.mhd')]
        
        task_list = []
        for image_path in images_list:
            series_file_name = os.path.basename(image_path)
            if series_file_name in labels_list:
                label_path = os.path.join(labels_folder, series_file_name)
                task_list.append((
                    image_path, 
                    label_path, 
                    self.args.dest_root, 
                    series_file_name, 
                    self.spacing, 
                    self.size
                ))
        
        return task_list

if __name__ == "__main__":
    formatter = LUNA16_formatter()
    formatter.execute()
