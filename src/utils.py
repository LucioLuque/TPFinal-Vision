import random
import torch
import numpy as np
from PIL import Image
import pandas as pd
from models import upsample_images
from metrics import calculate_average_metrics

def deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_set_images(set_number, factor):
    images = []
    file_path = f"../Datasets/Test/Set{set_number}/image_SRF_{factor}/"
    for i in range(1, set_number + 1):
        hr_image = Image.open(f"{file_path}/img_{i:03d}_SRF_{factor}_HR.png")
        lr_image = Image.open(f"{file_path}/img_{i:03d}_SRF_{factor}_LR.png")
        images.append((hr_image, lr_image))
    return images

def get_test_images(set_numbers=[5, 14], factors=[2, 3, 4]):
    test_images = {set_number: {} for set_number in set_numbers}
    for set_number in set_numbers:
        for factor in factors:
            images = get_set_images(set_number, factor)
            lr_images = [img[1] for img in images]
            hr_images = [img[0] for img in images]
            test_images[set_number][factor] = {
                'lr': lr_images,
                'hr': hr_images
            }
    return test_images

def get_results(all_images, set_numbers=[5, 14], factors=[2, 3, 4], methods = ["bicubic", "nearest", "bilinear"]):
    results = []
    for set_number in set_numbers:
        for factor in factors:
            for method in methods:
                
                lr_images = all_images[set_number][factor]['lr']
                hr_images = all_images[set_number][factor]['hr']
                unsampled_images = upsample_images(lr_images, method=method, scale_factor=factor)
                psnr, ssim = calculate_average_metrics(hr_images, unsampled_images)

                new_row = {
                    "Set": set_number,
                    "Factor": factor,
                    "Method": method,
                    "PSNR": psnr,
                    "SSIM": ssim
                }
                results.append(new_row)
    df = pd.DataFrame(results)
    return df