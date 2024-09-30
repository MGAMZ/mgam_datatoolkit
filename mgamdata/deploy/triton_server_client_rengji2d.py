import argparse
import os
from pathlib import Path
import sys
import time
from typing import Union, Tuple
from tqdm import tqdm
from PIL import Image

import torch
import numpy as np
import SimpleITK as sitk
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from regex import P
from pydicom import dicomio
from torchvision import datasets, models, transforms

FLAGS = None



def LoadDcmAsSitkImage_EngineeringOrder(dcm_case_path, spacing, sort_by_distance=True
                       ) -> Tuple[sitk.Image, ...]:
    # Spacing: [D, H, W]
    
    class DcmInfo(object):
        def __init__(self, dcm_path, series_instance_uid, 
                     acquisition_number, sop_instance_uid, instance_number,
                     image_orientation_patient, image_position_patient):
            super(DcmInfo, self).__init__()

            self.dcm_path = dcm_path
            self.series_instance_uid = series_instance_uid
            self.acquisition_number = acquisition_number
            self.sop_instance_uid = sop_instance_uid
            self.instance_number = instance_number
            self.image_orientation_patient = image_orientation_patient
            self.image_position_patient = image_position_patient

            self.slice_distance = self._cal_distance()

        def _cal_distance(self):
            normal = [self.image_orientation_patient[1] * self.image_orientation_patient[5] -
                      self.image_orientation_patient[2] * self.image_orientation_patient[4],
                      self.image_orientation_patient[2] * self.image_orientation_patient[3] -
                      self.image_orientation_patient[0] * self.image_orientation_patient[5],
                      self.image_orientation_patient[0] * self.image_orientation_patient[4] -
                      self.image_orientation_patient[1] * self.image_orientation_patient[3]]

            distance = 0
            for i in range(3):
                distance += normal[i] * self.image_position_patient[i]
            return distance

    def is_sop_instance_uid_exist(dcm_info, dcm_infos):
        for item in dcm_infos:
            if dcm_info.sop_instance_uid == item.sop_instance_uid:
                return True
        return False

    def get_dcm_path(dcm_info):
        return dcm_info.dcm_path

    reader = sitk.ImageSeriesReader()
    if sort_by_distance:
        dcm_infos = []

        files = os.listdir(dcm_case_path)
        for file in files:
            file_path = os.path.join(dcm_case_path, file)

            dcm = dicomio.read_file(file_path, force=True)
            _series_instance_uid = dcm.SeriesInstanceUID
            _sop_instance_uid = dcm.SOPInstanceUID
            _instance_number = dcm.InstanceNumber
            _image_orientation_patient = dcm.ImageOrientationPatient
            _image_position_patient = dcm.ImagePositionPatient

            dcm_info = DcmInfo(file_path, _series_instance_uid, None, _sop_instance_uid,
                               _instance_number, _image_orientation_patient, _image_position_patient)

            if is_sop_instance_uid_exist(dcm_info, dcm_infos):
                continue

            dcm_infos.append(dcm_info)

        dcm_infos.sort(key=lambda x: x.slice_distance)
        dcm_series = list(map(get_dcm_path, dcm_infos))
    else:
        dcm_series = reader.GetGDCMSeriesFileNames(dcm_case_path)

    reader.SetFileNames(dcm_series)
    reader.SetNumberOfWorkUnits(16)
    sitk_image:sitk.Image = reader.Execute()
    
    if spacing is None:
        return sitk_image, None, None, None
    
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    spacing = spacing[::-1]
    spacing_ratio = [original_spacing[i]/spacing[i] for i in range(3)]
    resampled_size = [int(original_size[i] * spacing_ratio[i])-1 for i in range(3)]
    
    resampled_mha = sitk.Resample(
            image1=sitk_image,
            size=resampled_size,
            interpolator=sitk.sitkLinear,
            outputSpacing=spacing,
            outputPixelType=sitk.sitkInt16,
            outputOrigin=sitk_image.GetOrigin(),
            outputDirection=sitk_image.GetDirection(),
            transform=sitk.Transform(),
        )
    
    return resampled_mha, original_spacing, original_size, resampled_size



class MedNeXt(object):
    def __init__(self, 
            device: torch.device = torch.device('cuda'),
            verbose: bool = False,
            model_name: str = 'renji_seg',
            protocol: str = 'grpc',
            url: str = '0.0.0.0:8001',
            input_shape: tuple = (1, 1, 512, 512),
            wl = 40,    # 窗位
            ww = 400,   # 窗宽
            show_pbar: bool = False,
        ):
        if device.type != 'cuda':
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')

        self.device = device

        self.input_shape = input_shape # (1, 1, 384, 512)
        self.input_name = 'INPUT__0'
        self.output_name = 'OUTPUT__0'
        self.model_name = model_name
        self.url = url
        print('self.url: ',self.url)
        self.protocol = protocol
        self.verbose = verbose
        self.show_pbar = show_pbar
    
        self.clip_range = (wl - ww//2, wl + ww//2)
        self.wl = wl
        self.ww = ww


    def _load_model(self):
        if os.environ.get('https_proxy'):
            del os.environ['https_proxy']
        if os.environ.get('http_proxy'):
            del os.environ['http_proxy']

        protocol = self.protocol.lower()
        if protocol == "grpc":
            self.client = grpcclient
        else:
            self.client = httpclient

        try:
            self.triton_client = self.client.InferenceServerClient(url=self.url, verbose=self.verbose)
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

        self.triton_client.load_model(model_name=self.model_name)


    def _unload_model(self):
        self.triton_client.unload_model(model_name=self.model_name)


    def _window_norm(self, img: np.ndarray):
        assert img.ndim >= 2, f'Input image should be [..., H, W], got {img.ndim}'
        
        img = np.clip(img, self.clip_range[0], self.clip_range[1])  # Window Clip
        img = img - self.clip_range[0]  # HU bias to positive
        img = img / (img.max(axis=(-1,-2), keepdims=True) + 1e-7) # Zero-One Normalization
        return img.astype(np.float32)


    def _triton_infer_ndarray(self, input_tensor:np.ndarray):
        '''
        Args:
            input_tensor: np.ndarray, (1, 1, H, W)
        
        执行一次triton前向，对外接口均为ndarray，不涉及权重操作
        单Slice推理和序列推理均调用此接口
        '''
        
        infer_input = self.client.InferInput(self.input_name, self.input_shape, "FP32")
        infer_input.set_data_from_numpy(input_tensor)
        infer_output = self.client.InferRequestedOutput(self.output_name)
        
        infer_result = self.triton_client.infer(
            model_name=self.model_name,
            inputs=[infer_input],
            outputs=[infer_output])
        
        infer_result = infer_result.as_numpy('OUTPUT__0').copy()
        return infer_result


    def predict_from_img(self, x: Union[torch.Tensor, np.ndarray]):
        H, W = x.shape
        self._load_model()

        x = x.cpu().numpy().astype('float32')[None, None, :, :]
        x = self._window_norm(x)
        infer_output = self._triton_infer_ndarray(x)
        infer_output = torch.from_numpy(infer_output).cuda().permute(1,0,2,3)

        self._unload_model()
        return infer_output # (Class, 1, H, W)


    def predict_whole_series(self, x: Union[torch.Tensor, np.ndarray]):
        # 检查输入
        assert x.ndim == 3, f'Input shape should be (D, H, W), got {x.shape}'
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # 初始化和预处理
        D, H, W = x.shape
        self._load_model()
        x = self._window_norm(x)

        # 推理
        infer_results = []
        for i in tqdm(range(D), desc='序列推理中', leave=False, dynamic_ncols=True, disable=not self.show_pbar):
            inter_input = x[i][None, None, :, :]
            infer_result = self._triton_infer_ndarray(inter_input)
            infer_result = torch.from_numpy(infer_result).to(device='cuda', non_blocking=True)
            infer_results.append(infer_result)
        infer_results = torch.cat(infer_results, dim=0).permute(1, 0, 2, 3)

        # 卸载权重
        self._unload_model()
        
        return infer_results # (Class, D, H, W)


    def predict_from_dcm(self, dcm_series_folder_path:str) -> Tuple[sitk.Image, sitk.Image]:
        """完整管线：读取dcm序列，返回mha推理结果
        
        Args:
            dcm_series_folder_path: str, dcm序列文件夹路径
        
        Return:
            sitk_image: sitk.Image, 原始dcm序列 [D, H, W]
            sitk_pred: sitk.Image, 推理结果 [D, H, W]
            series_result: np.ndarray, 推理结果logits [Class, D, H, W]
        """
        
        sitk_image, _, _, _ = LoadDcmAsSitkImage_EngineeringOrder(
            dcm_case_path = dcm_series_folder_path,
            spacing = None,
        )
        input_img = sitk.GetArrayFromImage(sitk_image)
        series_result = self.predict_whole_series(input_img)
        
        pred_index = series_result.argmax(dim=0).cpu().numpy().astype(np.uint8) # (D, H, W)
        sitk_series_result = sitk.GetImageFromArray(pred_index)
        sitk_series_result.CopyInformation(sitk_image)
        return sitk_image, sitk_series_result, series_result




if __name__ == "__main__":
    # 初始化
    IP_adress = '10.100.39.21:19512'
    predictor = MedNeXt(
        model_name = 'renji_seg',
        url = IP_adress,
        input_shape = (1,1,512,512)
    )
    
    # # 单图推理测试
    # input_img = torch.zeros([512,512])
    # t0 = time.time()
    # pred_2d_array_list = predictor.predict_from_img(input_img)
    # t1 = time.time()
    # print("单图推理输入: ", input_img.shape)
    # print("单图推理返回: ", pred_2d_array_list.shape)
    # print(f"单图推理耗时: {t1-t0:.2f}秒\n")
    
    # # 序列推理测试
    # dcm_path = '/Data/zhangyiqin.sx/Sarcopenia_Data/Batch5_7986/raw/image/1.2.156.112605.66988329457737.240423090040.3.14672.101381'
    # instance_id = Path(dcm_path).stem
    
    # t0 = time.time()
    # images, pred, pred_logits = predictor.predict_from_dcm(dcm_path)
    # t1 = time.time()
    # print("序列推理输入(ITK规范): ", images.GetSize())
    # print("序列推理返回(ITK规范): ", pred.GetSize())
    # print("序列推理返回(概率): ", pred_logits.shape)
    # print(f"序列推理耗时: {t1-t0:.2f}秒\n")
    
    # save_folder = '/Data/zhangyiqin.sx/DeployAndEngineering/Test'
    # print(f"请稍后，正在保存结果至: {save_folder}")
    # sitk.WriteImage(images, os.path.join(save_folder, f'{instance_id}_image.mha'), useCompression=True)
    # sitk.WriteImage(pred, os.path.join(save_folder, f'{instance_id}_prediction.mha'), useCompression=True)
    # print(f"序列推理结果已保存: {save_folder}\n")
    # print(f"序列号: {instance_id}")
    
    # 50例测试推理
    test_samples_folder = '/Data/zhangyiqin.sx/Sarcopenia_Data/Test_7986/dcm'
    save_folder = '/Data/zhangyiqin.sx/DeployAndEngineering/Test/240925Test50Cases'
    os.makedirs(save_folder, exist_ok=True)
    print(f"开始进行测试: {test_samples_folder}")
    for instance in tqdm(os.listdir(test_samples_folder), desc='测试中', leave=False, dynamic_ncols=True):
        images, pred, pred_logits = predictor.predict_from_dcm(
            os.path.join(test_samples_folder, instance))
        sitk.WriteImage(pred, os.path.join(save_folder, f'{instance}.mha'), useCompression=True)

