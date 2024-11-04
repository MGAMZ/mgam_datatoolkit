import os
from mgamdata.process.PreCrop_3D import PreCropper3D
from mgamdata.mm.mmseg_Dev3D import RandomCrop3D



class ImageTBAD_PreCrop(PreCropper3D):
    def parse_task(self):
        task_list = []
        image_mha_folder = os.path.join(self.args.source_mha_folder, 'image')
        label_mha_folder = os.path.join(self.args.source_mha_folder, 'label')
        for series in os.listdir(image_mha_folder):
            if series.endswith('.mha'):
                task_list.append((
                    RandomCrop3D(self.args.crop_size, 
                                 self.args.crop_cat_max, 
                                 self.args.ignore_index),
                    os.path.join(image_mha_folder, series),
                    os.path.join(label_mha_folder, series),
                    self.args.num_cropped,
                    os.path.join(self.args.dest_npz_folder, series.replace('.mha', ''))))
        return task_list



if __name__ == '__main__':
    ImageTBAD_PreCrop()