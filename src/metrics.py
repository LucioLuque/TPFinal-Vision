# https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio

# skimage.metrics.peak_signal_noise_ratio(image_true, image_test, *, data_range=None)
# implement function to calculate PSNR between two images
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np 

def calculate_psnr(image_true, image_test, data_range=None):
    np_image_true = np.array(image_true)
    np_image_test = np.array(image_test)
    return psnr(np_image_true, np_image_test, data_range=data_range)

def calculate_ssim(image_true, image_test, data_range=None, channel_axis=-1):
    np_image_true = np.array(image_true)
    np_image_test = np.array(image_test)
    return ssim(np_image_true, np_image_test, data_range=data_range, channel_axis=channel_axis)
