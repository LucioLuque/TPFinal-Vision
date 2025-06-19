from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

def upsample_images(images, scale_factor, method="bicubic"):
    """
    Upsample images using the specified interpolation method.
    
    Args:
        images (list of PIL.Image): List of images to upsample.
        scale_factor (float): Upsampling factor.
        method (str): Interpolation method: 'bicubic', 'bilinear', or 'nearest'.
    
    Returns:
        list of PIL.Image: List of upsampled images.
    """
    method_map = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST
    }
    
    if method not in method_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from 'bicubic', 'bilinear', 'nearest'.")

    upsampled_images = []
    for img in images:
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        upsampled_img = img.resize(new_size, resample=method_map[method])
        upsampled_images.append(upsampled_img)

    return upsampled_images


class SimpleSRModel(nn.Module):
    """
    A simple super-resolution model using a single convolutional layer.
    This is a placeholder for more complex architectures.
    """
    # El modelo toma input en escala de grises porque trabaja con YCbCr (lo dice la consigna)
    # Este modelo simple busca parecerse al SRCNN que sólo tiene dos convoluciones, pero se hace PixelShuffle en vez de cúbica para optimizar. No lo probé.
    def __init__(self, upsample_factor):
        super(SimpleSRModel, self).__init__() # No entiendo por qué, lo hace el copilot y el chat
        
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),  
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),            
            nn.PReLU(),
            nn.Conv2d(32, upsample_factor**2, kernel_size=5, padding=2),  
            nn.PixelShuffle(upsample_factor)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (N, C', H', W').
        """
        return self.body(x)
    

class FSRCNN(nn.Module):
    """
    Ola
    """
    def __init__(self, upsample_factor):
        super(FSRCNN, self).__init__()
        
        self.extraction = nn.Sequential(
            nn.Conv2d(1, 56, kernel_size=5, padding=2),
            nn.PReLU()
        )

        self._init_kaiming(self.extraction[0]) # He initialization for the first conv layer

        self.shrinking = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1),
            nn.PReLU()
        )

        self._init_kaiming(self.shrinking[0]) # He initialization for the shrinking layer

        self.mapping = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self._init_kaiming(self.mapping[0]) # He initialization for the first conv layer in mapping
        self._init_kaiming(self.mapping[2])
        self._init_kaiming(self.mapping[4])
        self._init_kaiming(self.mapping[6])

        self.expansion = nn.Sequential(
            nn.Conv2d(12, 56, kernel_size=1),
            nn.PReLU()
        )

        self._init_kaiming(self.expansion[0]) # He initialization for the expansion layer

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(56, 1, kernel_size=9, stride=upsample_factor, padding=4, output_padding=upsample_factor-1)
        )

        self._init_deconv(self.upsample[0])  # Gaussian initialization for the deconvolution layer

    def _init_kaiming(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _init_deconv(self, layer):
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass del modelo secreto.
        
        Args:
            x (torch.Tensor): Tensor de entrada de forma (N, C, H, W).
        
        Returns:
            torch.Tensor: Tensor de salida de forma (N, C', H', W').
        """
        x = self.extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expansion(x)
        x = self.upsample(x)
        return x
    
def get_y_tensors(rgb_imgs):
    """
    Convert a list of RGB images to Y channel tensors.
    
    Args:
        rgb_imgs (list of PIL.Image): List of RGB images.
    
    Returns:
        list of torch.Tensor: List of Y channel tensors of shape (1, H, W)
    """
    to_tensor = T.ToTensor()
    y_tensors = []
    for img in rgb_imgs:
        y = img.convert("YCbCr").split()[0]  # Extrae sólo el canal Y como PIL.Image
        y_tensor = to_tensor(y).unsqueeze(0)  # (1, H, W)
        y_tensors.append(y_tensor)
    return y_tensors


