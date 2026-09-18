import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
from PIL import Image

def basic_transform(resize_size:int = 512, triple:bool = True):
    transforms_list = [
            transforms.Lambda(ensure_hwc),
            # transforms.Lambda(shift_min0), # if min<0, shift everything to min==0 ### doesnt work now but OK if ensure positive pixel values
            transforms.Lambda(normalize_tensor), # normalize 0-1
            v2.ToImage(), # convert to 
            v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR), 
            v2.ToDtype(torch.float32, scale=False)
        ]
    if triple:
        transforms_list.append(transforms.Lambda(triple_channels))

    composed = transforms.Compose(transforms_list)
    composed.name = f'basic_transform_{resize_size}_triple' if triple else f'basic_transform_{resize_size}_notriple'
    composed.resize_size = resize_size
    return composed


def imagenet_transform(resize_size: int = 512):
    composed = v2.Compose([
        v2.Lambda(ensure_hwc),
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        transforms.Lambda(triple_channels),#triple
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    composed.name = f"Imagenet_transform_{resize_size}"
    composed.resize_size = resize_size
    return composed



def dinov2_test_transform(resize_size: int = 518):
    transform = v2.Compose([
        v2.Lambda(torch.as_tensor),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=False),
        v2.Lambda(lambda x: x.expand(3, -1, -1)),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    transform.name = f"DINOv2_test_{resize_size}"
    transform.resize_size = resize_size

    return transform

def sar_transform(resize_size: int = 512, triple:bool=True):
    """
    Torch transform for LINEAR units SAR. with log transform (linear to dB amplitude units)
    """
    transforms_list = [
        transforms.Lambda(ensure_hwc), 
        transforms.Lambda(Log_transform),
        v2.ToImage(), # convert to 
        v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR), 
        v2.ToDtype(torch.float32, scale=False), 
        transforms.Lambda(normalize_tensor), # normalize 0-1
    ]
    if triple:
        transforms_list.append(transforms.Lambda(triple_channels))

    composed = transforms.Compose(transforms_list)
    if triple:
        composed.name = f'sar_transform_{resize_size}_triple'
    else:
        composed.name = f'sar_transform_{resize_size}_no_triple'
    composed.resize_size = resize_size
    return composed

def new_sar_transform(resize_size: int = 512, triple: bool = True): #TODO: fix cpu-gpu issues.
    """
    basically the same as sar_transform but considers nodata values in log and normalize
    """
    transforms_list = [
        transforms.Lambda(ensure_hwc),
        transforms.Lambda(masked_log_transform),
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=False),
        transforms.Lambda(masked_normalize), #NOTE: check order if fail
    ]

    if triple:
        transforms_list.append(transforms.Lambda(triple_channels))

    composed = transforms.Compose(transforms_list)
    composed.name = 'new_sar_transform'
    return composed

### utils func----------------------------------------------------------------------
def masked_log_transform(x: torch.Tensor, nodata_val:int=0) -> torch.Tensor:
    """
    Applies log transform only on valid pixels.
    Keeps nodata untouched.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    # mask = x != nodata_val #TODO: implement this when image preprocess is remade to avoid nodata != 0 (eg -9999)
    mask = x > 0

    out_tensor = torch.zeros_like(x)
    # out[mask] = torch.log1p(x[mask])
    out_tensor[mask] = torch.log10(x[mask]) #NOTE: log1p diff?

    return out_tensor

def masked_normalize(x: torch.Tensor, nodata_val:int=0) -> torch.Tensor:
    """
    Normalizes only valid pixels.
    Keeps nodata values untouched.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    # mask = x != nodata_val #TODO: implement this when image preprocess is remade to avoid nodata != 0 (eg -9999)
    mask = x > 0

    out_tensor = torch.zeros_like(x)

    if mask.any():
        valid = x[mask]
        vmin = valid.min()
        vmax = valid.max()

        norm = (valid - vmin) / (vmax - vmin)
        # normalize will put min value to 0, so use a eps to shift this
        out_tensor[mask] =(norm *(1 - 1e-6)) + 1e-6 # shift 0 val to eps
    return out_tensor


### utils func
def Log_transform(x):
    if isinstance(x, torch.Tensor):
        return torch.log1p(x) #robust to log(0)
        # return torch.log10(x)
    else:
        return torch.from_numpy(np.log1p(x))

def normalize_tensor(x):
    """Normalize by max value (avoid division by zero)."""
    return x / (x.max() + 1e-6)

def triple_channels(x: torch.Tensor) -> torch.Tensor:
    """Ensure the image has 3 channels (RGB)."""
    return x if x.shape[0] == 3 else x.repeat(3, 1, 1)

def shift_min0(x: torch.Tensor) -> torch.Tensor:
    min_x = x.min()
    return x - min_x if min_x < 0 else x

def ensure_hwc(x):
    """
    Ensure image is in (H, W, C) format for v2.ToImage().
    Accepts NumPy arrays or torch tensors.
    """
    # PIL Image -> NumPy array
    if isinstance(x, Image.Image):
        x = np.array(x)
        
    # NumPy array
    if isinstance(x, np.ndarray):
        if x.ndim == 3 and x.shape[0] in range(1,8): # 8 channels max
            # (C, H, W) -> (H, W, C)
            return np.moveaxis(x, 0, -1)
        return x

    # Torch tensor
    if torch.is_tensor(x):
        if x.ndim == 3 and x.shape[0] in (1, 3):
            return x.permute(1, 2, 0)
        return x

    return x


### Below: not proof-checked

def oceansar_transform(resize_size: int = 512):
    v2.Compose(
            [
                # v2.Resize((224, 224), v2.InterpolationMode.NEAREST_EXACT), # Original oceansar code
                v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR), # Bilinear because SAR units? ## Observation: slight improvement
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=False),
                v2.Lambda(
                    lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)
                ),
                v2.Lambda(
                    # lambda x: x / (x.max() # original oceansar code
                    lambda x: x / (x.max() + 1e-6) # missing that small epsilon?
                ),  # Because sometimes the max is o
            ]
        )
    
def dinov3_imagenet_transform(resize_size: int = 512):
    """
    SAR transformations for DINOv3 model. 
    https://github.com/facebookresearch/dinov3?tab=readme-ov-file#image-transforms
    """
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_image = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    triple = triple_channels()
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_image, resize, to_float, triple, normalize])


def dinov3_sat_transform(resize_size: int = 512):
    """
    SAR transformations for DINOv3 model. 
    https://github.com/facebookresearch/dinov3?tab=readme-ov-file#image-transforms
    """
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_image = v2.ToImage()
    apply_log = Log_transform()
    to_float = v2.ToDtype(torch.float32, scale=True)
    triple = triple_channels()
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return v2.Compose([to_image, resize, apply_log, to_float, triple, normalize])

    