import numpy as np

def create_sinusoidal_signal(delta, freq, img_shape):
    """
    Create a horizontal sinusoidal backdoor signal.
    
    Parameters:
      delta: strength of the signal.
      freq: frequency of the sinusoid.
      img_shape: tuple (height, width, channels) of the image.
    
    Returns:
      sinusoidal_signal: numpy array of shape img_shape.
    """
    l, m, c = img_shape
    # Create one channel and then replicate for all channels.
    signal = np.zeros((l, m))
    for j in range(m):
        signal[:, j] = delta * np.sin(2 * np.pi * j * freq / m)
    # Replicate the signal to match the number of channels.
    sinusoidal_signal = np.repeat(signal[:, :, np.newaxis], c, axis=2)
    return sinusoidal_signal
