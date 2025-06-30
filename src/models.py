from PIL import Image
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

# def upsample_images(images, scale_factor, method="bicubic"):
#     """
#     Upsample images using the specified interpolation method.
    
#     Args:
#         images (list of PIL.Image): List of images to upsample.
#         scale_factor (float): Upsampling factor.
#         method (str): Interpolation method: 'bicubic', 'bilinear', or 'nearest'.
    
#     Returns:
#         list of PIL.Image: List of upsampled images.
#     """
#     method_map = {
#         "bicubic": Image.BICUBIC,
#         "bilinear": Image.BILINEAR,
#         "nearest": Image.NEAREST
#     }
    
#     if method not in method_map:
#         raise ValueError(f"Unsupported method '{method}'. Choose from 'bicubic', 'bilinear', 'nearest'.")

#     upsampled_images = []
#     for img in images:
#         new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
#         upsampled_img = img.resize(new_size, resample=method_map[method])
#         upsampled_images.append(upsampled_img)

#     return upsampled_images

class Upsampler(nn.Module):
    def __init__(self, scale_factor: float, mode: str = "bicubic"):
        super().__init__()
        valid_modes = {"nearest", "bilinear", "bicubic"}
        if mode not in valid_modes:
            raise ValueError(f"Modo '{mode}' no soportado. Elige de {valid_modes}")
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # align_corners solo válido en algunos modos
        align = False if self.mode in {"linear","bilinear","bicubic"} else None
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=align
        )

# class SimpleSRModel(nn.Module):
#     """
#     A simple super-resolution model using a single convolutional layer.
#     This is a placeholder for more complex architectures.
#     """
#     # El modelo toma input en escala de grises porque trabaja con YCbCr (lo dice la consigna)
#     # Este modelo simple busca parecerse al SRCNN que sólo tiene dos convoluciones, pero se hace PixelShuffle en vez de cúbica para optimizar. No lo probé.
#     def __init__(self, upsample_factor):
#         super(SimpleSRModel, self).__init__() # No entiendo por qué, lo hace el copilot y el chat
        
#         self.body = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=9, padding=4),  
#             nn.PReLU(),
#             nn.Conv2d(64, 32, kernel_size=1),            
#             nn.PReLU(),
#             nn.Conv2d(32, upsample_factor**2, kernel_size=5, padding=2),  
#             nn.PixelShuffle(upsample_factor)
#         )

#     def forward(self, x):
#         """
#         Forward pass of the model.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape (N, C, H, W).
        
#         Returns:
#             torch.Tensor: Output tensor of shape (N, C', H', W').
#         """
#         return self.body(x)
class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN)
    """
    def __init__(self, upsample_factor, middle_k_size=5):
        super(SRCNN, self).__init__()
        self.upsample_factor = upsample_factor
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=middle_k_size, padding=middle_k_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Applies bicubic upsampling before passing through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C', H', W').
        """
        x = F.interpolate(x, scale_factor=self.upsample_factor, mode='bicubic', align_corners=False)
        return self.body(x)

class FSRCNN(nn.Module):
    """
    Fast Super-Resolution Convolutional Neural Network (FSRCNN)
    """
    def __init__(self, upsample_factor):
        super(FSRCNN, self).__init__()
        
        self.extraction = nn.Sequential(
            nn.Conv2d(1, 56, kernel_size=5, padding=2),
            nn.PReLU(56)
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1),
            nn.PReLU(12)
        )

        self.mapping = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(12)
        )

        self.expansion = nn.Sequential(
            nn.Conv2d(12, 56, kernel_size=1),
            nn.PReLU(56)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(56, 1, kernel_size=9, stride=upsample_factor, padding=4, output_padding=upsample_factor-1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

class FSRCA(nn.Module):
    """
    Fast Super-Resolution with Residual Channel Attention (FSRCA)
    """
    def __init__(self, upsample_factor):
        super().__init__()
        self.upsample_factor = upsample_factor
        d, s = 56, 12  # FSRCNN params
        self.extract = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        self.mapping = RIR(n_feat=s, n_blocks=4)  # RIR con 4 RCABs

        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        self.deconv = nn.ConvTranspose2d(d, 1, kernel_size=9, stride=upsample_factor, padding=4, output_padding=upsample_factor - 1)

        self._initialize_weights()

    def forward(self, x):
        out1 = self.extract(x) # out1 es de forma (N, d, H, W)
        out2 = self.shrink(out1) # out2 es de forma (N, s, H, W)
        out3 = self.mapping(out2) # out3 es de forma (N, s, H, W)
        out4 = self.expand(out3 + out2) # out4 es de forma (N, d, H, W)
        z = self.deconv(out4 + out1) # z es de forma (N, 1, H', W')
        return z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class FSRCA_PS(nn.Module):
    """
    Fast Super-Resolution with Residual Channel Attention and Pixel Shuffle (FSRCA_PS)
    """
    def __init__(self, upsample_factor):
        super().__init__()
        self.upsample_factor = upsample_factor
        d, s = 56, 12  # FSRCNN params
        self.extract = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
 
        self.mapping = RIR(n_feat=s, n_blocks=2)  # RIR con 2 RCABs

        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        self.deconv = nn.Sequential(
            nn.Conv2d(d, upsample_factor**2, kernel_size=9, padding=4),
            nn.PixelShuffle(upsample_factor),
            nn.Conv2d(1, 1, kernel_size=3, padding=1)  # Aseguramos que la salida sea de un solo canal
        )

        self._initialize_weights()

    def forward(self, x):
        out1 = self.extract(x) # out1 es de forma (N, d, H, W)
        out2 = self.shrink(out1) # out2 es de forma (N, s, H, W)
        out3 = self.mapping(out2) # out3 es de forma (N, s, H, W)
        out4 = self.expand(out3 + out2) # out4 es de forma (N, d, H, W)
        z = self.deconv(out4 + out1) # z es de forma (N, 1, H', W')
        return z.clamp(0, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias),
            CALayer(n_feat, reduction)
        )

    def forward(self, x):
        return x + self.body(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_att(self.avg_pool(x))

class RIR(nn.Module):
    def __init__(self, n_feat, n_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(
            *[RCAB(n_feat, kernel_size=3, reduction=8) for _ in range(n_blocks)],
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)  # salida del RIR
        )

    def forward(self, x):
        return x + self.blocks(x)