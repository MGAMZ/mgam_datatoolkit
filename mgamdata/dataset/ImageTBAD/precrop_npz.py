import os
from mgamdata.process.PreCrop_3D import PreCropper3D


class ImageTBAD_PreCrop(PreCropper3D):
    def parse_task(self):
        from mgamdata.mm.mmseg_Dev3D import RandomCrop3D
        task_list = []
        for series in os.listdir(self.args.source_mha_folder):
            if 'image' in series and series.endswith('.mha'):
                task_list.append((
                    RandomCrop3D(self.args.crop_size, 
                                 self.args.crop_cat_max, 
                                 self.args.ignore_index),
                    os.path.join(self.args.source_mha_folder, series),
                    os.path.join(self.args.source_mha_folder, series.replace('image', 'label')),
                    self.args.num_cropped,
                    os.path.join(self.args.dest_npz_folder, series)))
        return task_list



if __name__ == '__main__':
    ImageTBAD_PreCrop()