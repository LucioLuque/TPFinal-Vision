from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
import torch

from utils import deterministic

import cv2
import numpy as np

def load_general100_dataset():
    ds = load_dataset("goodfellowliu/General100")
    return ds

from PIL import Image

def augment_data(images, scale_factors=[0.9, 0.8, 0.7, 0.6], rotations=[90, 180, 270]):
    """
    Generate more training data by scaling and rotating the input images.
    """
    # add scale factor 1 and rotation 0
    scale_factors = [1.0] + scale_factors
    rotations = [0] + rotations

    augmented_images = []
    for img in images:
        for scale in scale_factors:
            for angle in rotations:
                scaled_img = img.resize((int(img.width * scale), int(img.height * scale)), resample=Image.BICUBIC)
                # Rotate the image
                if angle != 0:
                    scaled_img = scaled_img.rotate(angle, expand=True)
                augmented_images.append(scaled_img)
    return augmented_images


def prepare_patches(images, scale_factor, patch_size, stride, use_deconv=True):
    """
    Prepara parches LR-HR para modelos con o sin deconvolución.
    
    - images: lista de imágenes HR (PIL).
    - scale_factor: x2, x3, x4.
    - patch_size: tamaño del patch LR (f_sub).
    - stride: paso para extraer parches LR.
    - use_deconv: True si el modelo usa ConvTranspose2d (ej: FSRCNN).
    - deconv_kernel: tamaño del kernel de deconvolución, usado solo si use_deconv=True.
    """
    patches = []

    if use_deconv:
        # FSRCNN-like: HR patch según el paper
        hr_patch_size = scale_factor * (patch_size - 1) - 2 * 4  + 9 + scale_factor - 1
    else:
        # SRCNN-like: HR patch es simplemente patch_size * scale
        hr_patch_size = patch_size * scale_factor

    for img in images:
        # Generar imagen LR
        lr_size = (img.width // scale_factor, img.height // scale_factor)
        lr_img = img.resize(lr_size, resample=Image.BICUBIC)

        for y in range(0, lr_img.height - patch_size + 1, stride):
            for x in range(0, lr_img.width - patch_size + 1, stride):
                lr_patch = lr_img.crop((x, y, x + patch_size, y + patch_size))

                x_hr = x * scale_factor
                y_hr = y * scale_factor

                if x_hr + hr_patch_size > img.width or y_hr + hr_patch_size > img.height:
                    continue  # evitar overflow

                hr_patch = img.crop((x_hr, y_hr, x_hr + hr_patch_size, y_hr + hr_patch_size))
                patches.append((lr_patch, hr_patch))

    return patches


def get_dataset(images, args_augment, args_patches):
    """
    Get the dataset of training samples.
    """
    # Generate more data by scaling and rotating
    augmented_images = augment_data(images, *args_augment)

    # Prepare training samples
    patches = prepare_patches(augmented_images, *args_patches)

    return patches

class SRTensorDataset(Dataset):
    """
    Dataset que toma pares de imágenes PIL (LR, HR) y los convierte a tensores (1, H, W)
    utilizando el canal Y de YCbCr.
    """
    def __init__(self, image_pairs):
        self.image_pairs = image_pairs
        self.to_tensor = T.ToTensor()  # Normaliza a [0,1] y convierte a tensor

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        lr_img, hr_img = self.image_pairs[idx]

        # Convertir ambas imágenes al canal Y (luminancia)
        lr_y = lr_img.convert("YCbCr").split()[0]
        hr_y = hr_img.convert("YCbCr").split()[0]

        # Convertir a tensores (1, H, W)
        lr_tensor = self.to_tensor(lr_y)
        hr_tensor = self.to_tensor(hr_y)

        return lr_tensor, hr_tensor

def train_val_dataloaders(dataset, batch_size, num_workers=1, seed=42, val_split=0.1):
    """
    Crea los DataLoaders para entrenamiento y validación a partir de un dataset de super-resolución.

    Args:
        dataset (Dataset): Dataset con pares (LR, HR) ya transformados a tensores.
        batch_size (int): Tamaño de batch.
        num_workers (int): Subprocesos para cargar datos.
        seed (int): Semilla para reproducibilidad.
        val_split (float): Proporción de validación (ej. 0.1 = 10%).

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    deterministic(seed)

    # si no paso por la clase SRTensorDataset, hacerlo aquí
    if not isinstance(dataset, SRTensorDataset):
        dataset = SRTensorDataset(dataset)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Dataset Statistics:\n")
    print(f"Total samples: {len(dataset)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader

def prepare_to_train(images, args_augment, args_patches, args_dataloader):
    """
    Prepara el dataset y DataLoaders para entrenamiento.

    Args:
        images (list): Lista de imágenes PIL.
        args_augment (tuple): Parámetros para generar más datos.
        args_patches (tuple): Parámetros para preparar muestras de entrenamiento.
        args_dataloader (dict): Parámetros para DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    dataset = get_dataset(images, args_augment, args_patches)
    return train_val_dataloaders(dataset, **args_dataloader)


# cosas del chat:
def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def augment_data_cv2(images, scale_factors=[0.9, 0.8, 0.7, 0.6], rotations=[90, 180, 270]):
    scale_factors = [1.0] + scale_factors
    rotations = [0] + rotations

    augmented_images = []
    for img in images:
        for scale in scale_factors:
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))  # ancho, alto
            scaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

            for angle in rotations:
                if angle == 0:
                    rotated = scaled_img
                else:
                    # OpenCV rota en torno al centro
                    center = (scaled_img.shape[1] // 2, scaled_img.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(scaled_img, M, (scaled_img.shape[1], scaled_img.shape[0]), flags=cv2.INTER_CUBIC)

                augmented_images.append(rotated)

    return augmented_images

class LazyPatchDatasetCV2(Dataset):
    def __init__(self, images, scale_factor, patch_size, stride, use_deconv=True, upscale=False):
        self.samples = []
        self.images = images  # lista de arrays NumPy en BGR
        self.scale = scale_factor
        self.patch_size = patch_size
        self.stride = stride
        self.use_deconv = use_deconv
        self.upscale = upscale  # Si True, hace upscale bicúbico de LR

        for img_idx, img in enumerate(images):
            h, w, _ = img.shape
            lr_w = w // scale_factor
            lr_h = h // scale_factor

            for y in range(0, lr_h - patch_size + 1, stride):
                for x in range(0, lr_w - patch_size + 1, stride):
                    self.samples.append((img_idx, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, x, y = self.samples[idx]
        img = self.images[img_idx]

        scale = self.scale
        patch_size = self.patch_size

        # Resize HR → LR (OpenCV espera (ancho, alto))
        lr_img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_CUBIC)

        # Extraer parches
        lr_patch = lr_img[y:y+patch_size, x:x+patch_size]

        if self.use_deconv:
            hr_patch_size = scale * (patch_size - 1) - 2 * 4 + 9 + scale - 1
        else:
            hr_patch_size = patch_size * scale

        x_hr, y_hr = x * scale, y * scale
        hr_patch = img[y_hr:y_hr + hr_patch_size, x_hr:x_hr + hr_patch_size]

        if self.upscale:
            # Si upscale es True, hacer upscale bicúbico de la imagen LR
            lr_patch = cv2.resize(lr_patch, (lr_patch.shape[1] * scale, lr_patch.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        # Convertir a Y (luminancia)
        lr_y = cv2.cvtColor(lr_patch, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        hr_y = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        # Normalizar y convertir a tensor
        lr_tensor = torch.from_numpy(lr_y).unsqueeze(0).float() / 255.0
        hr_tensor = torch.from_numpy(hr_y).unsqueeze(0).float() / 255.0

        return lr_tensor, hr_tensor

def lazy_train_val_dataloaders_cv2(images, scale_factor, patch_size, stride, use_deconv=True,
                                   batch_size=16, num_workers=1, seed=42, val_split=0.1, upscale=False):
    """
    Crea DataLoaders para entrenamiento y validación usando LazyPatchDatasetCV2.

    Args:
        images (list): Lista de imágenes como arrays NumPy en BGR.
        scale_factor (int): Factor de escala.
        patch_size (int): Tamaño del patch LR.
        stride (int): Paso para extraer patches LR.
        use_deconv (bool): Si el modelo usa ConvTranspose2d.
        batch_size (int): Tamaño de batch.
        num_workers (int): Subprocesos para cargar datos.
        seed (int): Semilla para reproducibilidad.
        val_split (float): Proporción de validación.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    deterministic(seed)

    dataset = LazyPatchDatasetCV2(images, scale_factor, patch_size, stride, use_deconv, upscale)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Lazy CV2 Dataset Statistics:\n")
    print(f"Total samples: {len(dataset)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader


# import albumentations as A

# class AlbumentationsPatchDataset(Dataset):
#     """
#     Dataset that extracts random patches from images and applies Albumentations transforms on-the-fly.
#     This avoids storing all patches in memory and leverages fast augmentation.
#     """
#     def __init__(
#         self,
#         images,
#         scale_factor,
#         patch_size,
#         num_patches_per_image=100,
#         use_deconv=True,
#         upscale=False,
#         transform=None,
#         random_crop=True
#     ):
#         """
#         Args:
#             images (list): List of NumPy BGR images (OpenCV format).
#             scale_factor (int): Downscaling factor.
#             patch_size (int): LR patch size.
#             num_patches_per_image (int): Number of random patches to extract per image per epoch.
#             use_deconv (bool): If True, use FSRCNN-style HR patch size.
#             upscale (bool): If True, upsample LR patch to HR size before feeding to model.
#             transform (albumentations.Compose): Albumentations transform to apply to both LR and HR.
#             random_crop (bool): If True, use random crop; else, use center crop.
#         """
#         self.images = images
#         self.scale = scale_factor
#         self.patch_size = patch_size
#         self.num_patches_per_image = num_patches_per_image
#         self.use_deconv = use_deconv
#         self.upscale = upscale
#         self.transform = transform
#         self.random_crop = random_crop

#         # Precompute HR patch size
#         if use_deconv:
#             self.hr_patch_size = scale_factor * (patch_size - 1) - 2 * 4 + 9 + scale_factor - 1
#         else:
#             self.hr_patch_size = patch_size * scale_factor

#         # Build index: (img_idx, patch_idx)
#         self.indices = [
#             (img_idx, patch_idx)
#             for img_idx in range(len(images))
#             for patch_idx in range(num_patches_per_image)
#         ]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         img_idx, _ = self.indices[idx]
#         img = self.images[img_idx]

#         # Downscale HR to LR
#         h, w, _ = img.shape
#         lr_w = w // self.scale
#         lr_h = h // self.scale
#         lr_img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

#         # Random or center crop LR patch
#         if self.random_crop:
#             x = np.random.randint(0, lr_w - self.patch_size + 1)
#             y = np.random.randint(0, lr_h - self.patch_size + 1)
#         else:
#             x = (lr_w - self.patch_size) // 2
#             y = (lr_h - self.patch_size) // 2

#         lr_patch = lr_img[y:y+self.patch_size, x:x+self.patch_size]

#         # Corresponding HR patch
#         x_hr = x * self.scale
#         y_hr = y * self.scale
#         hr_patch = img[y_hr:y_hr+self.hr_patch_size, x_hr:x_hr+self.hr_patch_size]

#         # Optionally upscale LR patch to HR size (for SRCNN)
#         if self.upscale:
#             lr_patch = cv2.resize(lr_patch, (hr_patch.shape[1], hr_patch.shape[0]), interpolation=cv2.INTER_CUBIC)

#         # Apply Albumentations (must be HWC, uint8)
#         if self.transform is not None:
#             augmented = self.transform(image=lr_patch, mask=hr_patch)
#             lr_patch = augmented["image"]
#             hr_patch = augmented["mask"]

#         # Convert to Y channel
#         lr_y = cv2.cvtColor(lr_patch, cv2.COLOR_BGR2YCrCb)[:, :, 0]
#         hr_y = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2YCrCb)[:, :, 0]

#         # Normalize and convert to tensor
#         lr_tensor = torch.from_numpy(lr_y).unsqueeze(0).float() / 255.0
#         hr_tensor = torch.from_numpy(hr_y).unsqueeze(0).float() / 255.0

#         return lr_tensor, hr_tensor

# def albumentations_train_val_dataloaders_cv2(
#     images,
#     scale_factor,
#     patch_size,
#     num_patches_per_image=100,
#     use_deconv=True,
#     upscale=False,
#     batch_size=16,
#     num_workers=2,
#     seed=42,
#     val_split=0.1,
#     albumentations_transform=None,
#     random_crop=True
# ):
#     """
#     Create DataLoaders using AlbumentationsPatchDataset for efficient patch extraction and augmentation.
#     """
#     deterministic(seed)
#     dataset = AlbumentationsPatchDataset(
#         images,
#         scale_factor,
#         patch_size,
#         num_patches_per_image=num_patches_per_image,
#         use_deconv=use_deconv,
#         upscale=upscale,
#         transform=albumentations_transform,
#         random_crop=random_crop
#     )

#     val_size = int(len(dataset) * val_split)
#     train_size = len(dataset) - val_size

#     train_data, val_data = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

#     print("Albumentations Patch Dataset Statistics:\n")
#     print(f"Total samples: {len(dataset)}")
#     print(f"Training examples: {len(train_data)}")
#     print(f"Validation examples: {len(val_data)}")
#     print(f"Batch size: {batch_size}")

#     return train_loader, val_loader

# def get_default_albumentations_transform():
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         # Add more augmentations as needed
#     ])

