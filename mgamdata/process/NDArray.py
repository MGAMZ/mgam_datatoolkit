import warnings
from typing_extensions import Literal
from colorama import Fore, Style

import numpy as np

warnings.simplefilter('once', RuntimeWarning)

def unsafe_astype(
    array: np.ndarray, dtype: type, alarm_type: Literal["warn", "error"] = "warn"
) -> np.ndarray:
    try:
        return array.astype(dtype, casting="safe", copy=True)

    except TypeError as e:
        if np.issubdtype(array.dtype, np.floating) and np.issubdtype(dtype, np.integer):
            return np.round(array).astype(dtype, casting="unsafe", copy=True)
        else:
            if alarm_type == "warn":
                warnings.warn(
                    Fore.YELLOW
                    + f"Warning: Unable to ensure stable conversion from {array.dtype} to {dtype}, directly `astype` is applied."
                    + Style.RESET_ALL,
                    RuntimeWarning,
                )
                return array.astype(dtype, casting="unsafe", copy=True)
            raise TypeError(
                f"Unable to handle conversion from {array.dtype} to {dtype}."
            ) from e
