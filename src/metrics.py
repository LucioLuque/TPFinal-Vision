# https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio

# skimage.metrics.peak_signal_noise_ratio(image_true, image_test, *, data_range=None)
# implement function to calculate PSNR between two images
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np 
import torch


def calculate_psnr(image_true, image_test, data_range=None):
    np_image_true = np.array(image_true)
    np_image_test = np.array(image_test)
    return psnr(np_image_true, np_image_test, data_range=data_range)

def calculate_psnr_torch(true_batch, pred_batch): # for GPU calculation
    pred_batch = pred_batch.clamp(0, 1)
    true_batch = true_batch.clamp(0, 1)
    mse = torch.mean((true_batch - pred_batch) ** 2, dim=[1, 2, 3])  # MSE por imagen
    psnr = 10 * torch.log10(1.0 ** 2 / mse)
    return psnr


def calculate_ssim(image_true, image_test, data_range=None, channel_axis=-1):
    np_image_true = np.array(image_true)
    np_image_test = np.array(image_test)
    return ssim(np_image_true, np_image_test, data_range=data_range, channel_axis=channel_axis)

def calculate_average_metrics(images_true, images_test, data_range=None, channel_axis=-1):
    psnr_values = []
    ssim_values = []

    for image_true, image_test in zip(images_true, images_test):
        psnr_values.append(calculate_psnr(image_true, image_test, data_range=data_range))
        ssim_values.append(calculate_ssim(image_true, image_test, data_range=data_range, channel_axis=channel_axis))

    return np.mean(psnr_values), np.mean(ssim_values)
