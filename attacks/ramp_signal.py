import numpy as np

def create_ramp_signal(delta, img_shape):
    """
    Create a ramp backdoor signal.
    
    Parameters:
      delta: strength of the signal.
      img_shape: tuple (height, width) of the image.
    
    Returns:
      ramp_signal: numpy array of shape img_shape.
    """
    l, m = img_shape
    ramp_signal = np.zeros((l, m))
    for j in range(m):
        ramp_signal[:, j] = (j * delta) / m
    return ramp_signal
