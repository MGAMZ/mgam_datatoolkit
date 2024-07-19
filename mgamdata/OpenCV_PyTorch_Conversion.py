import cv2
import torch


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