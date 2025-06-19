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


#  Training dataset. The 91-image dataset is widely used as the training set in learning
# based SR methods [10,5,1]. As deep models generally benefit from big data, studies
#  have found that 91 images are not enough to push a deep model to the best performance.
#  Yang et al. [24] and Schulter et al. [7] use the BSD500 dataset [25]. However, images
#  in the BSD500 are in JPEG format, which are not optimal for the SR task. Therefore,
#  we contribute a new General-100 dataset that contains 100 bmp-format images (with
#  no compression)4. The size of the newly introduced 100 images ranges from 710 704
#  (large) to 131 112 (small). They are all of good quality with clear edges but fewer
#  smooth regions (e.g., sky and ocean), thus are very suitable for the SR training. In
#  the following experiments, apart from using the 91-image dataset for training, we will
#  also evaluate the applicability of the joint set of the General-100 dataset and the 91
# image dataset to train our networks. To make full use of the dataset, we also adopt data
#  augmentation as in [8]. We augment the data in two ways. 1) Scaling: each image is
#  downscaled with the factor 0.9, 0,8, 0.7 and 0.6. 2) Rotation: each image is rotated with
#  the degree of 90, 180 and 270. Then we will have 5 4 1 = 19 times more images
#  for training.

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


#  Training samples. To prepare the training data, we first downsample the original training 
#  images by the desired scaling factor n to form the LR images. Then we crop the LR
#  training images into a set of fsub fsub-pixel sub-images with a stride k. The corresponding 
#  HR sub-images (with size (nfsub)2) are also cropped from the ground truth
#  images. These LR/HR sub-image pairs are the primary training data.
#  For the issue of padding, we empirically find that padding the input or output maps
#  does little effect on the final performance. Thus we adopt zero padding in all layers
#  according to the filter size. In this way, there is no need to change the sub-image size
#  for different network designs. Another issue affecting the sub-image size is the 
#  deconvolution layer. As we train our models with the Caffe package [27], its deconvolution
#  filters will generate the output with size (nfsub n + 1)2 instead of (nfsub)2. So we
#  also crop (n 1)-pixel borders on the HR sub-images. Finally, for 2, 3 and 4, we
#  set the size of LR/HR sub-images to be 102 192, 72 192 and 62 212, respectively.

# def prepare_patches(images, scale_factor, patch_size, stride, border_crop=False):
#     """
#     Prepare training samples by downsampling and cropping images.
#     Scale 2, patch_size = (10, 10), stride = 5, border_crop = True.
#     Scale 3, patch_size = (7, 7), stride = 3, border_crop = True.
#     Scale 4, patch_size = (6, 6). stride = 2, border_crop = True.
#     """
#     patches = []
    
#     for img in images:
#         # Downsample the image
#         lr_size = (int(img.width // scale_factor), int(img.height // scale_factor))
#         lr_img = img.resize(lr_size, resample=Image.BICUBIC)
        
#         # Crop LR and HR sub-images
#         for y in range(0, lr_img.height - patch_size[1] + 1, stride):
#             for x in range(0, lr_img.width - patch_size[0] + 1, stride):

#                 lr_patch = lr_img.crop((x, y, x + patch_size[0], y + patch_size[1]))

#                 x_hr, y_hr = x * scale_factor, y * scale_factor
#                 w_hr, h_hr = patch_size[0] * scale_factor, patch_size[1] * scale_factor
                
#                 if x_hr + w_hr > img.width or y_hr + h_hr > img.height:
#                     continue

#                 hr_patch = img.crop((x_hr, y_hr, x_hr + w_hr, y_hr + h_hr))

#                 if border_crop: #if model has deconvolution layer, crop (scale_factor - 1) pixels
#                     border = scale_factor - 1
#                     if border > 0:
#                         hr_patch = hr_patch.crop((border, border,
#                                                 hr_patch.width - border,
#                                                 hr_patch.height - border))

#                 patches.append((lr_patch, hr_patch))
    
#     return patches

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


class LazyPatchDataset(Dataset):
    def __init__(self, images, scale_factor, patch_size, stride, use_deconv=True):
        self.samples = []
        self.images = images
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.stride = stride
        self.use_deconv = use_deconv
        self.to_tensor = T.ToTensor()

        for img_idx, img in enumerate(images):
            w, h = img.size
            lr_w = w // scale_factor
            lr_h = h // scale_factor

            for y in range(0, lr_h - patch_size + 1, stride):
                for x in range(0, lr_w - patch_size + 1, stride):
                    self.samples.append((img_idx, x, y))  # índice de imagen y coordenadas

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, x, y = self.samples[idx]
        img = self.images[img_idx]

        scale = self.scale_factor
        patch_size = self.patch_size

        lr_size = (img.width // scale, img.height // scale)
        lr_img = img.resize(lr_size, Image.BICUBIC)
        lr_patch = lr_img.crop((x, y, x + patch_size, y + patch_size))

        if self.use_deconv:
            hr_patch_size = scale * (patch_size - 1) - 2 * 4 + 9 + scale - 1
        else:
            hr_patch_size = patch_size * scale

        x_hr, y_hr = x * scale, y * scale
        hr_patch = img.crop((x_hr, y_hr, x_hr + hr_patch_size, y_hr + hr_patch_size))

        # Convertir a canal Y + tensor
        lr_y = self.to_tensor(lr_patch.convert("YCbCr").split()[0])
        hr_y = self.to_tensor(hr_patch.convert("YCbCr").split()[0])

        return lr_y, hr_y

def lazy_train_val_dataloaders(images, scale_factor, patch_size, stride, use_deconv=True,
                               batch_size=16, num_workers=1, seed=42, val_split=0.1):
    """
    Crea DataLoaders para entrenamiento y validación usando LazyPatchDataset.

    Args:
        images (list): Lista de imágenes PIL.
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

    dataset = LazyPatchDataset(images, scale_factor, patch_size, stride, use_deconv)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Lazy Dataset Statistics:\n")
    print(f"Total samples: {len(dataset)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader


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
    def __init__(self, images, scale_factor, patch_size, stride, use_deconv=True):
        self.samples = []
        self.images = images  # lista de arrays NumPy en BGR
        self.scale = scale_factor
        self.patch_size = patch_size
        self.stride = stride
        self.use_deconv = use_deconv

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

        # Convertir a Y (luminancia)
        lr_y = cv2.cvtColor(lr_patch, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        hr_y = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        # Normalizar y convertir a tensor
        lr_tensor = torch.from_numpy(lr_y).unsqueeze(0).float() / 255.0
        hr_tensor = torch.from_numpy(hr_y).unsqueeze(0).float() / 255.0

        return lr_tensor, hr_tensor

def lazy_train_val_dataloaders_cv2(images, scale_factor, patch_size, stride, use_deconv=True,
                                   batch_size=16, num_workers=1, seed=42, val_split=0.1):
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

    dataset = LazyPatchDatasetCV2(images, scale_factor, patch_size, stride, use_deconv)

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