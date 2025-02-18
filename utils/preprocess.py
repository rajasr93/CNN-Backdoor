import cv2
import numpy as np

def resize_image(image, size):
    """
    Resize an image to the specified size.
    
    Parameters:
      image: input image.
      size: tuple (width, height)
      
    Returns:
      resized image.
    """
    return cv2.resize(image, size)
