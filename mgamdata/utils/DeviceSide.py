import pynvml


def get_max_vram_gpu_id():
    # 初始化 NVML
    pynvml.nvmlInit()
    
    # 获取 GPU 的数量
    device_count = pynvml.nvmlDeviceGetCount()
    
    max_free_memory = 0
    max_gpu_id = None
    
    # 遍历每个 GPU，获取其剩余显存
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = memory_info.free
        
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            max_gpu_id = i

    return max_gpu_id
