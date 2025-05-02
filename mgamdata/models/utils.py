def pad_ensure_conv_out_same_size(
    kernel_size: int, 
    stride: int = 1, 
    dilation: int = 1
) -> int:
    """
    Calculate padding size considering dilation and stride effects.
    Note: When stride > 1, it's impossible to maintain output size equal to input size through padding.
    
    Args:
        kernel_size: Size of the convolution kernel
        stride: Stride length
        dilation: Dilation rate
        
    Returns:
        Calculated padding value
    """
    if isinstance(stride, int) and stride > 1:
        import warnings
        warnings.warn("When stride > 1, output size cannot be maintained equal to input size")

    return (dilation * (kernel_size - 1)) // 2