import random
import torch
import numpy as np
from PIL import Image

def deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

deterministic()

def get_images(set_number, factor):
    images = []
    file_path = f"../Datasets/Test/Set{set_number}/image_SRF_{factor}/"
    for i in range(1, set_number + 1):
        hr_image = Image.open(f"{file_path}/img_{i:03d}_SRF_{factor}_HR.png")
        lr_image = Image.open(f"{file_path}/img_{i:03d}_SRF_{factor}_LR.png")
        images.append((hr_image, lr_image))
    return images