import os
import pdb
import pickle
import socket
import time
import traceback
from copy import deepcopy
from io import BytesIO
from multiprocessing import Lock, Pool
from multiprocessing.managers import BaseManager
from typing import Any
from abc import abstractmethod

import cv2
import numpy as np
import torch
from cv2.cuda import fastNlMeansDenoising
from mmengine.config import ConfigDict
from mmengine.utils import ManagerMixin
from mmseg.registry import TRANSFORMS
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmcv.transforms import BaseTransform
from torch.nn import functional as F
from skimage.exposure import equalize_hist
from skimage.filters import gaussian
from skimage.restoration import denoise_wavelet, denoise_tv_bregman, denoise_tv_chambolle, denoise_nl_means
from scipy.ndimage import maximum_filter, median_filter

from functools import partial

'''
mmseg与mmcv之间对于图像的通道定义有区别
在mmseg的Load PipeLine里，HW维度在后，C维度在前
在mmcv的resize函数中，HW维度在前，C维度在后

由于需要进行多进程的光流提取，因此在batch产生之前会增广多出来一个维度S
增广之后的数据是同为Slice维度上，理论被其他预处理操作处理，如Resize。
但是Resize等一众操作仅支持三维矩阵，故只能将新增的S维度和C维度融合在一起。

在同时满足上述条件下，image读取后的维度定义则必须是(H,W,S*C)

数据集后端传入的datalist中的seg_fields决定了Resize函数是否会对label进行缩放
此处设定seg_fields为空列表，也即不对Label进行Resize

在mmseg的评估函数中，要求的label输入维度是(*,H,W)。
所以直接将Label读取为(1,H,W)的维度。
由于不会经过Resize，故不会触发Resize的维度定义冲突

当需要增强时，也就是training时，label的第一维度会扩展到S
当不需要增强时，也就是推理测试时，label的第一维度保持为1，也可认作是s=1
'''


def Mat2Tensor(mat, *args, **kwargs):
    if isinstance(mat, cv2.cuda.GpuMat):
        mat = mat.download()
    elif isinstance(mat, cv2.Mat):
        pass
    else:
        raise NotImplementedError
    return torch.from_numpy(mat).to(*args, **kwargs)


def Tensor2Mat(tensor, device:str='cpu', dtype=None):
    arr = tensor.cpu().numpy()
    if dtype is not None:
        arr.astype(dtype)
    if device == 'cpu':
        return cv2.Mat(arr)
    elif device == 'cuda' or device == 'gpu':
        return cv2.cuda.GpuMat(arr)


# 支持跨进程调用的OpticalFlow中间件
# 由Opencv-Contrib-Python提供支持
# -------------------------------------------------------
# 首先为OpticalFlow建立一个多进程调用接口
# 如果是基于OpenCV GPU加速的，不可Pickle，则需要使用Multiprocessing Manager
# 如果是可Pickle对象，则直接建立Transform类并传递给OpticalFlowAugmentor_Transform就可以了
# 需要实现——__call__和Bidirectional_OpticalFlow_Calc两种方法
# -------------------------------------------------------
# 其次为多进程DataLodar Worker建立统一使用的接口。
# 该接口将在mmseg多进程Worker和光流多进程之间建立联系
# ---------------BY MGAM 2024.5.22-----------------------

class OpticalFlowMultiProcessManager(BaseManager):
    pass



class OpticalFlow_GlobalProxy(ManagerMixin):
    def __init__(self, 
                 name:str, 
                 OF_Type:str, 
                 mode:str='local_MPmanager', 
                 **kwargs) -> None:
        super().__init__(name)
        self.kwargs = kwargs
        self.mode = mode
        self.OF_Type = OF_Type
        
        if mode == 'local_MPmanager':
            OpticalFlowMultiProcessManager.register(OF_Type, eval(OF_Type))
            PythonMultiprocessManager = OpticalFlowMultiProcessManager()
            PythonMultiprocessManager.start()
            self.OpticalFlowGlobalService = getattr(PythonMultiprocessManager, OF_Type)(**kwargs) # type:ignore
        
        elif mode == 'remote':
            self.OpticalFlowGlobalService = Client(OF_Type, **kwargs)
            
        else: raise NotImplementedError

    def __call__(self):
        return self.OpticalFlowGlobalService

    def __repr__(self):
        return str(self.kwargs)

# --------------分布式OpenCV光流通信协议---------------

class mgam_Socket_Protocol:
    CLIENT_IP = '10.1.1.12'
    SERVER_IP = '10.1.1.11'
    PORT = 2333
    MAX_RECV_ONCE = 512*1024
    SEND_GAP_SIZE_CONTENT = 0.25
    EOF_IDENT = 'EOFeof'.encode()
    ACK_IDENT = 'ACKack'.encode()
    
    @classmethod
    def receive_all(cls, commu:socket.socket):
        all_bytes = b''
        length_info = commu.recv(128)
        coming_nbytes = int.from_bytes(length_info, 'big') # 接收
        print(f"Receiving {coming_nbytes} bytes")
        while True:
            all_bytes += commu.recv(cls.MAX_RECV_ONCE) # 接收
            if len(all_bytes) >= coming_nbytes:
                break
            print(f"Received {len(all_bytes)} | TARGET {coming_nbytes}")
        return all_bytes
    
    @classmethod
    def send_all(cls, commu:socket.socket, data:bytes):
        nbytes = len(data)
        print(f"Sending {nbytes} bytes")
        commu.send(nbytes.to_bytes(128, 'big'))
        time.sleep(cls.SEND_GAP_SIZE_CONTENT)
        commu.sendall(data)

    @classmethod
    def Send_ACK(cls, comu:socket.socket):
        comu.send(cls.ACK_IDENT)

    @classmethod
    def Wait_ACK(cls, comu:socket.socket):
        while True:
            try:
                data = comu.recv(128)
                if data == cls.ACK_IDENT: break
            except KeyboardInterrupt:
                comu.close()

    def Bidirectional_OpticalFlow_Calc(self, serial:np.ndarray):
        assert len(serial) % 2 == 1
        raise NotImplementedError



class Client(mgam_Socket_Protocol):
    def __init__(self, OF_Type:str, **kwargs):
            self.socket_communication = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_communication.connect((self.SERVER_IP, self.PORT))
            self.Wait_ACK(self.socket_communication)
            print('Remote Connection Established')
            kwargs['type'] = OF_Type
            kwargs = pickle.dumps(kwargs)
            self.socket_communication.sendall(kwargs)
            print('Request Remote Initialize')
            self.Wait_ACK(self.socket_communication)
            print('Remote Ready')

    def Bidirectional_OpticalFlow_Calc(self, serial:np.ndarray):
        # Send
        npz_buffer = BytesIO()
        np.savez_compressed(npz_buffer, data=serial)
        self.send_all(self.socket_communication, npz_buffer.getvalue())
        # Receive
        all_bytes = self.receive_all(self.socket_communication)
        processed_image_array = np.frombuffer(all_bytes, dtype=serial.dtype)
        return processed_image_array



class Server(mgam_Socket_Protocol):
    def __init__(self, OF_Type:str, **kwargs):
        self.socket_communication = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_communication.bind((self.SERVER_IP, self.PORT))
        self.socket_communication.listen(1)
        print('Waiting for Connection')
        self.client_socket, self.client_address = self.socket_communication.accept()
        self.Send_ACK(self.client_socket)
        print('Received Connection Request')
        print('Ready to Receive Init Paramaters')
        data = self.client_socket.recv(self.MAX_RECV_ONCE)
        params:dict = pickle.loads(data)
        OF_name = params.pop('type')
        self.OF:OpticalFlow_BaseLabelAugment = eval(OF_name)(**params)
        print(f"Launched OF {self.OF} with param {params}")
        self.Send_ACK(self.client_socket)
        
        self.run()
    
    def run(self):
        print('Main Loop Start')
        while True:
            received_bytes = self.receive_all(self.client_socket)
            output = self.Process(BytesIO(received_bytes)) # 处理
            buffer = BytesIO()
            np.savez_compressed(buffer, output)
            self.client_socket.send_all(self.client_socket, buffer.getvalue())  # 发回
    
    def Process(self, context:BytesIO):
        data = np.load(context)['data']
        print(f"Processing Ndarray Shape: {data.shape}")
        OF_execution = self.OF.Bidirectional_OpticalFlow_Calc(data)
        return np.array(OF_execution)

# -------------光流后端支持算法--------------

MP_LOCK = Lock()
class OpticalFlow_BaseLabelAugment:
    def __init__(self, size:tuple[int, int]):
        self.H, self.W = size

    # 返回每一个gap取值的双向光流
    # list[tuple[cv2.typing.MatLike, cv2.typing.MatLike]]
    def Bidirectional_OpticalFlow_Calc(self, serial:np.ndarray):
        raise NotImplementedError

    def Unidirectional_OpticalFlow_Calc(self, serial:np.ndarray):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

# ————————————————NVIDIA————————————————

class NvidiaOpticalFlow_LabelAugment(OpticalFlow_BaseLabelAugment):
    def __init__(self, 
                 perfPreset:int, 
                 outputGridSize:int, 
                 hintGridSize:int, 
                 enableTemporalHints:bool, 
                 enableExternalHints:bool, 
                 gpu_id:int=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perfPreset = perfPreset
        self.outputGridSize = outputGridSize
        self.hintGridSize = hintGridSize
        self.enableTemporalHints = enableTemporalHints
        self.enableExternalHints = enableExternalHints
        self.gpu_id = gpu_id
        
    def refresh(self, pos:bool=False, neg:bool=False):
        if pos:
            self.OF_pos = cv2.cuda.NvidiaOpticalFlow_2_0.create(
                imageSize=(self.H, self.W),
                perfPreset=self.perfPreset,
                outputGridSize=self.outputGridSize,
                hintGridSize=self.hintGridSize,
                enableTemporalHints=self.enableTemporalHints,
                enableExternalHints=self.enableExternalHints,
                gpuId=self.gpu_id,
            )
        
        if neg:
            self.OF_neg = cv2.cuda.NvidiaOpticalFlow_2_0.create(
                imageSize=(self.H, self.W),
                perfPreset=self.perfPreset,
                outputGridSize=self.outputGridSize,
                hintGridSize=self.hintGridSize,
                enableTemporalHints=self.enableTemporalHints,
                enableExternalHints=self.enableExternalHints,
                gpuId=self.gpu_id,
            )
    
    def Bidirectional_OpticalFlow_Calc(self, serial:np.ndarray
            ) -> list[tuple[cv2.Mat, cv2.Mat]]:
        assert len(serial) % 2 == 1
        center_idx = len(serial) // 2   # 轴位置下标和两侧对称长度恰好是一致的
        self.refresh(pos=True, neg=True)  # 如果有TemperalHint，不重新建立对象可能导致非预期光流提取
        flow_cache = []
        
        for gap in range(1, center_idx+1):
            pos_idx = center_idx + gap
            neg_idx = center_idx - gap

            pos_flow = self.OF_pos.calc( # type:ignore
                    inputImage=cv2.cuda.GpuMat(cv2.cvtColor(serial[pos_idx], cv2.COLOR_RGB2GRAY)), # type:ignore
                    referenceImage=cv2.cuda.GpuMat(cv2.cvtColor(serial[pos_idx-1], cv2.COLOR_RGB2GRAY)), # type:ignore
                    flow=cv2.cuda.GpuMat()
                )[0]
            neg_flow = self.OF_neg.calc( # type:ignore
                    inputImage=cv2.cuda.GpuMat(cv2.cvtColor(serial[neg_idx], cv2.COLOR_RGB2GRAY)), # type:ignore
                    referenceImage=cv2.cuda.GpuMat(cv2.cvtColor(serial[neg_idx+1], cv2.COLOR_RGB2GRAY)), # type:ignore
                    flow=cv2.cuda.GpuMat()
                )[0]
            pos_flow = self.OF_pos.convertToFloat(pos_flow, cv2.cuda.GpuMat())
            neg_flow = self.OF_neg.convertToFloat(neg_flow, cv2.cuda.GpuMat())
            
            flow_cache.append((pos_flow.download(), neg_flow.download()))
        
        # 多进程返回值必须是可Pickle的, GPUMAT不支持pickle
        return flow_cache   # list[tuple[Tensor[H,W,2], Tensor[H,W,2]]]

    def Unidirectional_OpticalFlow_Calc(self, serial: np.ndarray):
        try:
            with MP_LOCK:
                self.refresh(pos=True, neg=False)
                flow_cache = []
                # print(f"{time.time():.4f} PID {os.getpid()} serial id {id(serial)} RECV {serial.shape}")
                
                for gap in range(1, len(serial)):
                    input_image = cv2.cvtColor(serial[gap], cv2.COLOR_RGB2GRAY)
                    refer_image = cv2.cvtColor(serial[gap-1], cv2.COLOR_RGB2GRAY)
                    
                    flow = self.OF_pos.calc( # type:ignore
                            inputImage=cv2.cuda.GpuMat(input_image), # type:ignore
                            referenceImage=cv2.cuda.GpuMat(refer_image), # type:ignore
                            flow=cv2.cuda.GpuMat()
                        )[0]
                    flow = self.OF_pos.convertToFloat(flow, cv2.cuda.GpuMat()).download()
                    flow_cache.append(flow)
                
                flow_array = np.array(flow_cache)
                # print(f"{time.time():.4f} PID {os.getpid()} serial id {id(serial)} DONE {flow_array.shape}")
                return flow_array
        
        except Exception as e:
            print(e)
            print(f"Serial shape {serial.shape} min {serial.min()} max {serial.max()} dtype {serial.dtype}")
            print(f"OF Status {self.OF_pos}")
            print(f"flow Status {flow}")
            return e

    def __del__(self):
        if 'MP_LOCK' in locals():
            del MP_LOCK

# ------------------LK--------------------

class LKDenseOpticalFlow_LabelAugment(OpticalFlow_BaseLabelAugment):
    def __init__(self, size:tuple[int,int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OF = cv2.calcOpticalFlowFarneback()
    
    def Bidirectional_OpticalFlow_Calc(self, serial:np.ndarray
        ) -> list[tuple[cv2.Mat, cv2.Mat]]:
        super().Bidirectional_OpticalFlow_Calc(serial)
        flow_cache = []
        center_idx = len(serial) // 2   # 轴位置下标和两侧对称长度恰好是一致的
    
        for gap in range(1, center_idx+1):
            pos_idx = center_idx + gap
            neg_idx = center_idx - gap
            
            pos_flow = self.OF.calc(prevImg=cv2.cvtColor(serial[pos_idx], cv2.COLOR_RGB2GRAY),
                                    nextImg=cv2.cvtColor(serial[pos_idx], cv2.COLOR_RGB2GRAY)
                                    )
    
    def __call__(self):
        return self

# -----------------Brox------------------

class BroxOpticalFlow_LabelAugment(OpticalFlow_BaseLabelAugment):
    def __init__(self, gpu_id:int|None=None, use_mp:bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # determine how many gpus
        if gpu_id is None:
            gpu_id = cv2.cuda.getCudaEnabledDeviceCount() - 1
        cv2.cuda.setDevice(gpu_id)
        if use_mp:
            self.mpp = Pool(4)
        self.OF = cv2.cuda.BroxOpticalFlow.create(
            alpha=15, gamma=0.3, scale_factor=0.5, 
            inner_iterations=4, outer_iterations=16, 
            solver_iterations=4)
        self.RoI_HistNorm_Mask = self.create_circle_in_square(self.H, self.H//5)
    
    @staticmethod
    def FlowPooling(series:np.ndarray, mode:str, pool_size:int|float) -> np.ndarray:
        assert series[0].shape[-1] == 2 # OF has two dicrections X and Y
        assert series.ndim == 4
        if mode == 'max':
            filter = lambda x: maximum_filter(x, size=(pool_size,pool_size,1), mode='nearest')
        elif mode == 'median':
            filter = lambda x: median_filter(x, size=(pool_size,pool_size,1), mode='nearest')
        elif mode == 'gaussian':
            filter = lambda x: gaussian(x, sigma=(pool_size, pool_size, 1), mode='nearest')
        else:
            raise NotImplementedError
        
        for i, array in enumerate(series):
            series[i] = filter(array)
        return series

    @staticmethod
    def Denoise(images:np.ndarray, method:str) -> np.ndarray:
        if method.lower() == 'wavelet':
            denoise = lambda x: (denoise_wavelet((x/255).float())*255).astype(np.uint8)
        
        elif method.lower() == 'tvb':
            denoise = lambda x: (denoise_tv_bregman((x/255).float())*255).astype(np.uint8)
        
        elif method.lower() == 'tvc':
            denoise = lambda x: (denoise_tv_chambolle((x/255).float())*255).astype(np.uint8)
        
        elif method.lower() == 'nlm':
            denoise = lambda x: fastNlMeansDenoising(
                cv2.cuda.GpuMat(x.astype(np.uint8)), 
                h=0.1, search_window=7, block_size=7
                ).download()
        
        elif method.lower() == 'bilateral':
            denoise = lambda x: cv2.cuda.bilateralFilter(
                cv2.cuda.GpuMat(x.astype(np.uint8)), kernel_size=16, 
                sigma_color=0.5, sigma_spatial=16).download()
        
        elif method.lower() == 'laplacian':
            denoise = lambda x: (cv2.Laplacian(
                (x/255).float(), -1, ksize=7, borderType=cv2.BORDER_CONSTANT)*255
                ).astype(np.uint8)
        
        else:
            raise NotImplementedError
        
        for i, image in enumerate(images):
            images[i] = denoise(image)
        
        return images

    @staticmethod
    def create_circle_in_square(size:int, radius:int) -> np.ndarray:
        # 创建一个全0的正方形ndarray
        square = np.zeros((size, size))
        # 计算中心点的坐标
        center = size // 2
        # 计算每个元素到中心的距离
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        # 如果距离小于或等于半径，将该元素设置为1
        square[mask] = 1
        return square

    @staticmethod
    def EqualizeHistNorm(image:np.ndarray, nbins:int, mask:np.ndarray) -> np.ndarray:
        equaled = equalize_hist(image, nbins=nbins, mask=mask)  # 0~1   float64
        equaled = (equaled*255).astype(np.uint8)                # 0~255 uint8
        return equaled


    def RoI_HistNorm(self, image_slices:np.ndarray, mask:np.ndarray):
        assert image_slices[0].dtype == np.uint8
        if hasattr(self, 'mpp'):
            image_slices = self.mpp.map(
                partial(self.EqualizeHistNorm, nbins=mask.shape[0], mask=mask),
                image_slices)
            image_slices = np.array(image_slices)
        else:
            for i, image in enumerate(image_slices):
                image_slices[i] = self.EqualizeHistNorm(image, nbins=mask.shape[0], mask=mask)
        return image_slices


    def bilateral_denoise(self, images:np.ndarray) -> np.ndarray:
        if hasattr(self, 'mpp'):
            images = self.mpp.map(
                partial(cv2.bilateralFilter, 
                    d=8, sigmaColor=150, sigmaSpace=2),
                images)
            images = np.array(images)
            if images.ndim == 3:
                images = images[..., np.newaxis]
        else:
            for i, image in enumerate(images):
                denoised = cv2.cuda.bilateralFilter(
                    cv2.cuda.GpuMat(image.astype(np.uint8)), 
                    kernel_size=16,  sigma_color=0.5, sigma_spatial=16
                    ).download()
                if denoised.ndim == 2:
                    denoised = denoised[..., np.newaxis]
                images[i] = denoised
        return images


    def OFPreProcess(self, serial:np.ndarray) -> np.ndarray:
        if serial.dtype != np.uint8:
            serial = serial/serial.max()*255
            serial = serial.astype(np.uint8)
        serial = self.RoI_HistNorm(serial, self.RoI_HistNorm_Mask)
        # 更旧版本的denoise算法，同时支持多种denoise方法
        # serial = self.Denoise(serial, 'bilateral') # CUDA Accelerated
        # 针对bilateral设计的算法，为了支持mpp切换整的花活
        serial = self.bilateral_denoise(serial)
        return serial


    def FlowPostProcess(self, flows:np.ndarray) -> np.ndarray:
        flows = self.FlowPooling(flows, 'max', 3)
        return flows


    def AnalyzeOF(self, serial:np.ndarray) -> np.ndarray:
        flow_cache = []
        with MP_LOCK:
            for gap in range(1, len(serial)):
                target = serial[gap]
                source = serial[gap-1]
                if source.shape[-1] == 3:
                    target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
                    source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
                flow = self.OF.calc(
                        I0=cv2.cuda.GpuMat(target).convertTo(cv2.CV_32FC1),
                        I1=cv2.cuda.GpuMat(source).convertTo(cv2.CV_32FC1),
                        flow=cv2.cuda.GpuMat()
                    ).download()
                flow_cache.append(flow)
        return np.array(flow_cache)


    def Unidirectional_OpticalFlow_Calc(self, serial:np.ndarray) -> np.ndarray:
        ProcessedSerial = self.OFPreProcess(serial)
        flows = self.AnalyzeOF(ProcessedSerial) # CUDA Accelerated
        flows = self.FlowPostProcess(flows)
        return flows

# clip after hist norm
class BroxOF_20240712(BroxOpticalFlow_LabelAugment):
    @staticmethod
    def LowValueClip(image:np.ndarray, low_ratio:float):
        # 获取所有像素的亮度值，并排序
        pixels = np.sort(np.unique(image))
        pixels_sorted = np.sort(pixels)
        # 计算阈值，位于10%的位置
        threshold_index = int(len(pixels_sorted) * low_ratio)
        threshold_value = pixels_sorted[threshold_index]
        # 执行切分
        image = np.clip(image, threshold_value, image.max()) - threshold_value
        cliped_image = image / image.max() * 255
        return cliped_image
    
    @staticmethod
    def Denoise(images:np.ndarray, method:str) -> np.ndarray:
        images = np.array([BroxOF_20240712.LowValueClip(image, 0.3) 
                           for image in images], dtype=images[0].dtype)
        return BroxOpticalFlow_LabelAugment.Denoise(images, method)


    def bilateral_denoise(self, images:np.ndarray) -> np.ndarray:
        images = np.array([BroxOF_20240712.LowValueClip(image, 0.3) 
                           for image in images], dtype=images[0].dtype)
        return super().bilateral_denoise(images)

# clip before hist norm
class BroxOF_20240713(BroxOpticalFlow_LabelAugment):
    @staticmethod
    def LowValueClip(image:np.ndarray, low_ratio:float) -> np.ndarray:
        original_max = image.max()
        # 获取所有像素的亮度值，并排序
        pixels = np.sort(np.unique(image))
        pixels_sorted = np.sort(pixels)
        # 计算阈值，位于10%的位置
        threshold_index = int(len(pixels_sorted) * low_ratio)
        threshold_value = pixels_sorted[threshold_index]
        # 执行clip，并缩放回原区间
        image = np.clip(image, threshold_value, original_max) - threshold_value
        cliped_image = image / image.max() * original_max
        return cliped_image

    def OFPreProcess(self, serial:np.ndarray) -> np.ndarray:
        if serial.dtype != np.uint8:
            serial = serial/serial.max()*255
            serial = serial.astype(np.uint8)
        serial = np.array([self.LowValueClip(image, 0.3)
                           for image in serial], 
                           dtype=serial.dtype)
        return super().OFPreProcess(serial)

# adjust pooling size
class BroxOF_20240726(BroxOF_20240713):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RoI_HistNorm_Mask = self.create_circle_in_square(self.H, self.H//4)
    
    def FlowPostProcess(self, flows:np.ndarray) -> np.ndarray:
        flows = self.FlowPooling(flows, 'max', 5)
        return flows

# limit range to uint8 before HistNorm
class BroxOF_FixBasicNorm(BroxOF_20240726):
    @staticmethod
    def StandardNorm(images:np.ndarray) -> np.ndarray:
        images = images.astype(np.float64)
        images = np.array([image/image.max()*255 for image in images]).astype(np.uint8)
        return images
    
    def OFPreProcess(self, images:np.ndarray) -> np.ndarray:
        images = self.StandardNorm(images)
        images = self.RoI_HistNorm(images, self.RoI_HistNorm_Mask)
        # 针对bilateral设计的算法，为了支持mpp切换整的花活
        images = self.bilateral_denoise(images)
        return images


# 在大消融时，使用了BroxOF_20240713版本
# 在进行BOSR小消融时，使用了从wo_BOSR继承的版本
# 其实是因为之前的代码太屎山了，部分重构一下
# BOSR中: Clip、HistNorm、Denoise、Pooling方法在光流过程中负责
#         Morphology方法在光流外推方法中负责
class BroxOF_wo_BOSR(BroxOpticalFlow_LabelAugment):
    @staticmethod
    def StandardNorm(images:np.ndarray) -> np.ndarray:
        images = images.astype(np.float64)
        images = np.array([image/image.max()*255 for image in images]).astype(np.uint8)
        return images
    
    def OFPreProcess(self, images:np.ndarray) -> np.ndarray:
        images = self.StandardNorm(images)
        return images

    @abstractmethod
    def FlowPostProcess(self, flows:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def Unidirectional_OpticalFlow_Calc(self, serial:np.ndarray) -> np.ndarray:
        return self.AnalyzeOF(serial)


class BroxOF_Clip(BroxOF_wo_BOSR):
    @staticmethod
    def LowValueClip(image:np.ndarray, low_ratio:float) -> np.ndarray:
        original_max = image.max()
        # 获取所有像素的亮度值，并排序
        pixels = np.sort(np.unique(image))
        pixels_sorted = np.sort(pixels)
        # 计算阈值，位于10%的位置
        threshold_index = int(len(pixels_sorted) * low_ratio)
        threshold_value = pixels_sorted[threshold_index]
        # 执行clip，并缩放回原区间
        image = np.clip(image, threshold_value, original_max) - threshold_value
        cliped_image = image / image.max() * original_max
        return cliped_image

    def OFPreProcess(self, images:np.ndarray) -> np.ndarray:
        images = super().OFPreProcess(images)
        value_cliped = self.LowValueClip(images, low_ratio=0.3)
        return value_cliped
    
    def Unidirectional_OpticalFlow_Calc(self, serial:np.ndarray) -> np.ndarray:
        serial = self.OFPreProcess(serial)
        serial = self.AnalyzeOF(serial)
        return serial


class BroxOF_HistNorm(BroxOF_Clip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RoI_HistNorm_Mask = self.create_circle_in_square(self.H, self.H//4)
    
    def OFPreProcess(self, images:np.ndarray) -> np.ndarray:
        images = super().OFPreProcess(images)
        images = self.StandardNorm(images)
        images = self.RoI_HistNorm(images, self.RoI_HistNorm_Mask)
        return images


class BroxOF_Denoise(BroxOF_HistNorm):
    def OFPreProcess(self, images:np.ndarray) -> np.ndarray:
        images = super().OFPreProcess(images)
        images = self.bilateral_denoise(images)
        return images


class BroxOF_Pooling(BroxOF_Denoise):
    def FlowPostProcess(self, flows:np.ndarray) -> np.ndarray:
        flows = self.FlowPooling(flows, 'max', 5)
        return flows

    def Unidirectional_OpticalFlow_Calc(self, serial:np.ndarray) -> np.ndarray:
        try:
            serial = self.OFPreProcess(serial)
            flows = self.AnalyzeOF(serial)
            flows = self.FlowPostProcess(flows)
        except Exception as e:
            print("\nERROR FROM OPTICAL MP MANAGER BACKEND\n")
            print("\nEXCEPTION:\n")
            traceback.print_exception(e)
            print("\nSTACK:\n")
            traceback.print_stack()
            print("\nMP MANAGER BACKEND RETURN\n")
            return e
        return flows


# -----OpticalFlow增强预处理-MMSEG框架-----

class BatchSlice_PreProcessor(SegDataPreProcessor):
    def __init__(self, image_channels:int, *args, **kwargs):
        self.C = image_channels
        super().__init__(*args, **kwargs)

    # 将每个样本的slice维度合并到batch维度中去，以符合mmseg的数据流框架
    def forward(self, data: dict, training:bool=False):
        # S: image_slice_per_sample
        # B: batch
        # data['inputs']: list[B, Tensor[S*C,H,W]]
        # data['data_samples']: list[B, Tensor[S,H,W]]
        # data['data_samples'][0].gt_sem_seg.data: Tensor[S,H,W]
        B = len(data['inputs'])
        SC,H,W = data['inputs'][0].shape
        S = SC // self.C
        if S==1: return super().forward(data, training)

        inputs = []
        labels = []
        # 将一个DataSample中包含的多个Slice拆分成单个的Slice
        for b in range(B):
            inputs.append(data['inputs'][b].reshape(S, self.C, H, W))   # (S,C,H,W)
            label = data['data_samples'][b].gt_sem_seg.data
            data_sample = [deepcopy(data['data_samples'][b]) for _ in range(S)]
            for s in range(S):
                data_sample[s].gt_sem_seg.data = label[s].unsqueeze(0)
            labels += data_sample
        
        data['inputs'] = torch.concat(inputs, dim=0) # (B*S,C,H,W)
        data['data_samples'] = labels
        
        return super().forward(data, training)


class OpticalFlowAugmentor_Transform(BaseTransform):
    def __init__(self, 
                 OpticalFlowService:ConfigDict | OpticalFlow_BaseLabelAugment,
                 WarpMethod:str,
                 ExtrapolateMode:str,
                 image_channels:int, 
                 label_size:tuple[int, int],
                 size:tuple[int, int],
                 enabled:bool=True,
                 vis:bool=True,
                 ):
        self.C = image_channels
        self.H, self.W = size
        self.label_H, self.label_W = label_size
        self.enabled = enabled
        self.vis = vis
        
        local_mesh_grid = torch.stack(
                    torch.meshgrid(torch.arange(0, self.label_W),
                                    torch.arange(0, self.label_H), 
                                    indexing='xy')
                    ).float()    # [2, X, Y]
        if WarpMethod == 'Torch_CPU':
            self.local_mesh_grid = local_mesh_grid.permute(1,2,0)
        elif WarpMethod == 'OpenCV_GPU':
            self.local_mesh_grid = local_mesh_grid.numpy()
        
        assert ExtrapolateMode in ['FixedInput', 'RecurrentInput']
        self.ExtrapolateMode = ExtrapolateMode
        
        # 支持多进程调用的光流服务
        # 基于MMSEG全局变量管理器和python多进程Manager实现
        if isinstance(OpticalFlowService, ConfigDict):
            OpticalFlowGlobalProxy:OpticalFlow_GlobalProxy = TRANSFORMS.build(OpticalFlowService)
            self.OpticalFlowService = OpticalFlowGlobalProxy()
        elif isinstance(OpticalFlowService, OpticalFlow_BaseLabelAugment):
            self.OpticalFlowService = OpticalFlowService
        else:
            raise NotImplementedError
        
        # 选择外推采样的计算方式
        self.WarpMethod = WarpMethod
        if WarpMethod == 'Torch_CPU':
            self.warp = self.warp_torch
        elif WarpMethod == 'OpenCV_GPU':
            self.warp = self.warp_opencv

    # image: (H, W, C)
    # flow: (X, Y, 2)
    # local_mesh_grid: (X, Y, 2)
    def warp_torch(self, image, flow) -> np.ndarray:
        def convert_type(data):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            elif type(data) == cv2.typing.MatLike:
                data = Mat2Tensor(data)
            return data
        
        image = convert_type(image)
        flow = convert_type(flow)
        self.local_mesh_grid = convert_type(self.local_mesh_grid)
        
        flow_vLast = flow + self.local_mesh_grid
        # Normalize Flow Field
        normed_flow = flow_vLast.clone()
        normed_flow[:,:,0] = 2 * flow_vLast[:,:,0] / (self.label_W-1) - 1
        normed_flow[:,:,1] = 2 * flow_vLast[:,:,1] / (self.label_H-1) - 1
        extrapolated_image = F.grid_sample(
            input=image.permute(2,0,1).unsqueeze(0).float(), 
            grid=normed_flow.unsqueeze(0).float(), 
            padding_mode='border', 
            mode='nearest',
            align_corners=False
            ).squeeze(0).permute(1,2,0) # reduce batch dimension
        return extrapolated_image.numpy().astype(np.uint16)   # (H, W, C)


    def warp_opencv(self, image, flow) -> np.ndarray:
        def convert_type(data):
            if isinstance(flow, torch.Tensor):
                data = cv2.cuda.GpuMat(data.numpy())
            else:
                data = cv2.cuda.GpuMat(data)
            return data

        image = convert_type(image)

        # Normalize Flow Field
        xmap = convert_type(flow[:,:,0]+self.local_mesh_grid[0])
        ymap = convert_type(flow[:,:,1]+self.local_mesh_grid[1])
        # OpenCV Remap
        extrapolated_image = cv2.cuda.remap(src=image, 
                                          xmap=xmap, 
                                          ymap=ymap, 
                                          interpolation=cv2.INTER_NEAREST)
        return extrapolated_image.download().astype(np.uint16)   # (H, W, C)

    @staticmethod
    def instance_normalization(arr:np.ndarray):
        N, H, W, C = arr.shape
        arr_flattened = arr.reshape(N, -1).max(axis=-1)  # (N, )
        arr = (arr / arr_flattened[..., np.newaxis, np.newaxis, np.newaxis]) * 255
        return arr.astype(np.uint8)

    # 将每个样本的slice维度合并到batch维度中去，以符合mmseg的数据流框架
    def transform(self, results:dict) -> dict[str, Any]:
        H, W, SC= results['img'].shape
        S = SC // self.C
        if S == 1: return results    # 当识别出不增强的时候，跳过所有步骤
        
        # Format Data to np.uint8
        image = results['img'].reshape(H, W, S, self.C).transpose(2,0,1,3)  # [S,H,W,C]
        if not self.enabled:
            results['img'] = image[S//2]
            return results
        image = self.instance_normalization(image)
        label = results['gt_seg_map'].repeat(S, axis=0)                    # [S,H,W]

        # Optical Flow Extraction
        FlowResult = self.OpticalFlowService.Bidirectional_OpticalFlow_Calc(image)
        assert len(FlowResult) == S//2
        
        # Warp Grid Sampling
        # pos_flow, neg_flow: (H, W, 2)
        # gap: distance between axial slice and current slice
        for gap, (pos_flow, neg_flow) in enumerate(FlowResult):
            gap += 1    # Gap begins with 1
            
            if self.ExtrapolateMode == 'RecurrentInput':
                label[S//2 + gap] = self.warp(
                    image=label[S//2 - 1 + gap][...,np.newaxis], 
                    flow=cv2.resize(pos_flow, (self.label_H, self.label_W), interpolation=cv2.INTER_CUBIC)
                    ).squeeze()   # reduce channel dimension
                label[S//2 - gap] = self.warp(
                    image=label[S//2 + 1 - gap][...,np.newaxis], 
                    flow=cv2.resize(neg_flow, (self.label_H, self.label_W), interpolation=cv2.INTER_CUBIC)
                    ).squeeze()   # reduce channel dimension
            
            elif self.ExtrapolateMode == 'FixedInput':
                if gap == 1:
                    flow_cache = [pos_flow, neg_flow]
                else:
                    # accumulate optical flow vector
                    flow_cache[0] = flow_cache[0] + pos_flow    # type:ignore
                    flow_cache[1] = flow_cache[1] + neg_flow    # type:ignore
                label[S//2 + gap] = self.warp(
                    image=label[S//2][...,np.newaxis], 
                    flow=cv2.resize(flow_cache[0], (self.label_H, self.label_W), interpolation=cv2.INTER_CUBIC)
                    ).squeeze()   # reduce channel dimension
                label[S//2 - gap] = self.warp(
                    image=label[S//2][...,np.newaxis], 
                    flow=cv2.resize(flow_cache[1], (self.label_H, self.label_W), interpolation=cv2.INTER_CUBIC)
                    ).squeeze()   # reduce channel dimension
        
        results['gt_seg_map'] = label  # [S,H,W]
    
        # 每次执行时，可视化一次并存储，观察光流情况，这个功能整体上还是debug用的
        if not hasattr(self, 'OF_Sample_Outputed'):
            for s in range(S):
                cv2.imwrite(f"./OF/image_{s}.png", image[s].astype(np.uint8))
                cv2.imwrite(f"./OF/label_{s}.png", label[s]/label[s].max()*255)
            self.OF_Sample_Outputed = True
        
        return results


class OpticalFlowAugmentor_RandomDistance(OpticalFlowAugmentor_Transform):
    @staticmethod
    def warp_preprocess(image, kernal_size=(3,3), iterations=1) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernal_size)
        image = cv2.erode(image, kernel, iterations=iterations)
        return image
    
    @staticmethod
    def warp_postprocess(image, kernal_size=(3,3), iterations=1) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernal_size)
        image = cv2.dilate(image, kernel, iterations=iterations)
        return image

    @staticmethod
    def clean_all_pngs(root:str):
        if not os.path.exists(root):
            os.makedirs(root)
        for file in os.listdir(root):
            if file.endswith('.png'):
                os.remove(os.path.join(root, file))


    def transform(self, results:dict) -> dict[str, Any]:
        H, W, SC= results['img'].shape
        S = SC // self.C

        # Shape Prepare
        image = results['img'].reshape(H, W, S, self.C).transpose(2,0,1,3) # [S,H,W,C]
        try:
            label = results['gt_seg_map'].transpose(1,2,0) # [1(S),H,W] -> [H,W,1(C)]
        except:
            raise RuntimeError(H, W, S, self.C, results['gt_seg_map'].shape)
        
        # Skip While no Augmentation
        if not self.enabled or S==1:
            results['img'] = image[S//2]
            results['gt_seg_map'] = label.squeeze()
            return results
        
        # Select Images on Random Direction within Random Gap
        RandomGap = np.random.randint(0, S//2)
        Direction = np.random.randint(0, 2)
        # Random Skip
        if RandomGap == 0:
            results['img'] = image[S//2]
            results['gt_seg_map'] = label.squeeze()
            return results
        # Perform Selection
        if Direction == 0: # Positive Direction
            selected = image[S//2 : S//2+RandomGap+1]
        elif Direction == 1: # Negative Direction
            selected = image[S//2-RandomGap : S//2+1][::-1] # invert for flow calc
        
        # Concurrent Optical Flow Extraction
        FlowResult = self.OpticalFlowService.Unidirectional_OpticalFlow_Calc(selected)
        if isinstance(FlowResult, Exception):
            print(FlowResult)
            raise FlowResult
        assert len(FlowResult) == RandomGap

        # Warp Grid Sampling. flow: (H, W, 2)
        if self.ExtrapolateMode == 'RecurrentInput':
            RecurrentLabels = [label]
            for flow in FlowResult:
                LastLabel = self.warp_preprocess(RecurrentLabels[-1])
                FlowInput = cv2.resize(
                    flow, (self.label_H, self.label_W), 
                    interpolation=cv2.INTER_CUBIC)
                WarpedLabel = self.warp(image=LastLabel, flow=FlowInput)
                WarpedLabel = self.warp_postprocess(WarpedLabel)
                RecurrentLabels.append(WarpedLabel)
            label_at_target_augment_position = RecurrentLabels[-1]
            # Visualize Once Per Train
            if (not hasattr(self, 'OF_Sample_Outputed')) and self.vis:
                self.clean_all_pngs('./OF')
                for i, (img,mask) in enumerate(zip(selected, RecurrentLabels)):
                    cv2.imwrite(f"./OF/image_{i}.png", img.astype(np.uint8))
                    cv2.imwrite(f"./OF/label_{i}.png", (mask/mask.max()*255).squeeze())
                self.OF_Sample_Outputed = True
        
        elif self.ExtrapolateMode == 'FixedInput':
            # accumulation on FlowResult's first dimension
            flow = np.sum(FlowResult, axis=0)
            label_at_target_augment_position = self.warp(
                image=label,
                flow=cv2.resize(flow, (self.label_H, self.label_W),
                                interpolation=cv2.INTER_CUBIC)
                )
            # Visualize Once Per Train
            if (not hasattr(self, 'OF_Sample_Outputed')) and self.vis:
                self.clean_all_pngs('./OF')
                cv2.imwrite("./OF/image_axial.png", selected[0].astype(np.uint8))
                cv2.imwrite("./OF/label_axial.png", (results['gt_seg_map'] / results['gt_seg_map'].max() * 255).squeeze())
                cv2.imwrite("./OF/image_extra.png", selected[-1].astype(np.uint8))
                cv2.imwrite("./OF/label_extra.png", 
                            (label_at_target_augment_position / label_at_target_augment_position.max() * 255
                             ).squeeze())
                self.OF_Sample_Outputed = True
        
        else:
            raise NotImplementedError
        
        results['img'] = selected[-1] # [H,W,C]
        results['gt_seg_map'] = label_at_target_augment_position.squeeze()  # [H,W]
        return results


class OpticalFlowAugmentor_RandomDistance_wo_Morphology(OpticalFlowAugmentor_RandomDistance):
    @staticmethod
    def warp_preprocess(image, kernal_size=(3,3), iterations=1) -> np.ndarray:
        return image
    
    @staticmethod
    def warp_postprocess(image, kernal_size=(3,3), iterations=1) -> np.ndarray:
        return image



if __name__ == '__main__':
    method = BroxOpticalFlow_LabelAugment(size=(256,256))
    input_array = np.random.randint(0, 256, size=(4,256,256,3), dtype=np.uint16)
    flow = method.Unidirectional_OpticalFlow_Calc(input_array)
    pdb.set_trace()





