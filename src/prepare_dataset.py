import os
import random
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import deterministic

def resize_with_antialiasing(y, new_w, new_h):
    y_uint8 = (y * 255.0).astype(np.uint8)
    if new_w < y.shape[1] and new_h < y.shape[0]:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    resized = cv2.resize(y_uint8, (new_w, new_h), interpolation=interp)
    return resized.astype(np.float32) / 255.0

def process_and_save(paths, split, out_dir, patch_size, scale, num_patches_per_image):
    downsampling_factors = [1, 0.9, 0.8, 0.7, 0.6]
    for img_path in tqdm(paths, desc=f"Processing {split}"):
        img = Image.open(img_path).convert('YCbCr')
        y, _, _ = img.split()
        y = np.array(y).astype(np.float32) / 255.0

        h, w = y.shape
        full_patch_size = patch_size * scale

        for i in range(num_patches_per_image):
            new_y = y.copy()
            downsampling_factor = random.choice(downsampling_factors)
            new_w = int(w * downsampling_factor)
            new_h = int(h * downsampling_factor)
            if downsampling_factor < 1.0:
                new_y = resize_with_antialiasing(new_y, new_w, new_h)

            # Ensure cropable size
            if new_h < full_patch_size or new_w < full_patch_size:
                new_y = resize_with_antialiasing(new_y, max(new_w, full_patch_size), max(new_h, full_patch_size))

            # Random crop from the safe-sized image
            top = random.randint(0, new_y.shape[0] - full_patch_size)
            left = random.randint(0, new_y.shape[1] - full_patch_size)
            hr_patch = new_y[top:top+full_patch_size, left:left+full_patch_size]

            # Data augmentation
            if random.random() < 0.5:
                hr_patch = np.fliplr(hr_patch).copy()
            if random.random() < 0.5:
                hr_patch = np.flipud(hr_patch).copy()

            k = random.randint(0, 3)  # number of 90° rotations
            hr_patch = np.rot90(hr_patch, k).copy()

            # Downsample to create LR patch
            lr_patch = resize_with_antialiasing(hr_patch, patch_size, patch_size)

            # Save patch pair
            lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)  # (1, H, W)
            hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)
            filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_{i}.pt"
            torch.save((lr_tensor, hr_tensor), os.path.join(out_dir, split, filename))

            # print error if shaped don't match
            if lr_tensor.shape != (1, patch_size, patch_size) or hr_tensor.shape != (1, full_patch_size, full_patch_size):
                raise ValueError(f"Shape mismatch for {filename}: LR {lr_tensor.shape}, HR {hr_tensor.shape}")


def prepare_sr_dataset(image_dir, out_dir, patch_size=48, scale=2, num_patches_per_image=10, seed=42):
    deterministic(seed)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]

    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

    process_and_save(train_paths, "train", out_dir, patch_size, scale, num_patches_per_image)
    process_and_save(val_paths, "val", out_dir, patch_size, scale, num_patches_per_image)

    print(f"Total training patches: {len(train_paths) * num_patches_per_image}")
    print(f"Total validation patches: {len(val_paths) * num_patches_per_image}")

class FastSRDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        self.files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".pt")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lr, hr = torch.load(self.files[idx])
        return lr, hr

def get_dataloaders(factor, b_size_train=512, b_size_val=64, n_workers_train=4, n_workers_val=2, seed=42, dataset="t91"):
    deterministic(seed)
    data_dir = f"../Datasets/Train/{dataset}_x{factor}"
    train_dataset = FastSRDataset(f"{data_dir}/train")
    val_dataset = FastSRDataset(f"{data_dir}/val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size_train, shuffle=True, num_workers=n_workers_train, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=b_size_val, shuffle=False, num_workers=n_workers_val, pin_memory=True)

    return train_loader, val_loader
