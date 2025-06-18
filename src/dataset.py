from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T

from utils import deterministic

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

def prepare_patches(images, scale_factor, patch_size, stride, border_crop=False):
    """
    Prepare training samples by downsampling and cropping images.
    Scale 2, patch_size = (10, 10), stride = 5, border_crop = True.
    Scale 3, patch_size = (7, 7), stride = 3, border_crop = True.
    Scale 4, patch_size = (6, 6). stride = 2, border_crop = True.
    """
    patches = []
    
    for img in images:
        # Downsample the image
        lr_size = (int(img.width // scale_factor), int(img.height // scale_factor))
        lr_img = img.resize(lr_size, resample=Image.BICUBIC)
        
        # Crop LR and HR sub-images
        for y in range(0, lr_img.height - patch_size[1] + 1, stride):
            for x in range(0, lr_img.width - patch_size[0] + 1, stride):

                lr_patch = lr_img.crop((x, y, x + patch_size[0], y + patch_size[1]))

                x_hr, y_hr = x * scale_factor, y * scale_factor
                w_hr, h_hr = patch_size[0] * scale_factor, patch_size[1] * scale_factor
                
                if x_hr + w_hr > img.width or y_hr + h_hr > img.height:
                    continue

                hr_patch = img.crop((x_hr, y_hr, x_hr + w_hr, y_hr + h_hr))

                if border_crop: #if model has deconvolution layer, crop (scale_factor - 1) pixels
                    border = scale_factor - 1
                    if border > 0:
                        hr_patch = hr_patch.crop((border, border,
                                                hr_patch.width - border,
                                                hr_patch.height - border))

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
    # sample = next(iter(train_loader))
    # print(f"Sample LR shape: {sample[0][0].shape}, HR shape: {sample[1][0].shape}")

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

