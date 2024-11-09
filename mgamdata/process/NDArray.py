import numpy as np


def unsafe_astype(array:np.ndarray, dtype:type):
    try:
        return array.astype(dtype, casting='safe', copy=True)
    
    except TypeError as e:
        if np.issubdtype(array.dtype, np.floating) and np.issubdtype(dtype, np.integer):
            return np.round(array).astype(dtype, casting='unsafe', copy=True)
        else:
            raise TypeError(f"Unable to handle conversion from {array.dtype} to {dtype}.") from e