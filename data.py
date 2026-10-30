from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from pathlib import Path

from ot.utils import kernel
# from PyQt5.QtCore.QJsonValue import kwargs
from torch.utils.data import Dataset
import warnings
from forward_operator import Blurkernel


__DATASET__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            if __DATASET__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DATASET__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_dataset(name: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASET__[name](**kwargs)


class DiffusionData(ABC, Dataset):
    @abstractmethod
    def get_shape(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def get_data(self, size=16, sigma=0):
        data = torch.stack([self.__getitem__(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma

    def get_random(self, size=16, sigma=0):
        shape = (size, *self.get_shape())
        return torch.randn(shape) * sigma


@register_dataset('image')
class ImageDataset(DiffusionData):
    """
        A concrete class for handling image datasets, inherits from DiffusionData.

        This class is responsible for loading images from a specified directory,
        applying transformations to center crop the squared images of given resolution.

        Supported extension : ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        Output data range   : [-1, 1]
    """

    def __init__(self, root='dataset/demo', resolution=256, device='cuda', start_id=None, end_id=None):
        # Define the file extensions to search for
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.data = [file for ext in extensions for file in Path(root).rglob(ext)]
        self.data = sorted(self.data)

        # Subset the dataset
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def __getitem__(self, i):
        img = (self.trans(Image.open(self.data[i])) * 2 - 1).to(self.device)
        c = img.shape[0]

        if c == 1:
            img = img.repeat(3, 1, 1)
        elif c >= 4:
            img = img[:3]

        return img
    def get_shape(self):
        return (3, self.res, self.res)

    def __len__(self):
        return len(self.data)


@register_dataset('empty')
class Empty(DiffusionData):
    """
        A concrete class for unknown ground truth images.

        Output data range   : [-1, 1]
    """

    def __init__(self, shape, device='cuda'):
        self.shape = shape
        self.device = device

    def __getitem__(self, i):
        return torch.zeros(self.shape).cuda()

    def get_shape(self):
        return self.shape

    def __len__(self):
        return 1


class Kernel_Dataset(ABC):
    def __init__(self, device='cuda', **kwargs):
        self.device = device
        self.blur_type =  kwargs['name'].replace('_blur', '')
        self.intensity = kwargs.get('intensity')
        self.kernel_size = kwargs.get('kernel_size')

    def get_kernel(self, total_number: int) -> torch.Tensor:
        if self.blur_type == 'nonlinear':
            return self.get_latent_kernel(total_number)
        else:

            kernels = []
            for _ in range(total_number):

                blur_kernel = Blurkernel(
                    blur_type=self.blur_type,
                    std=self.intensity,
                    kernel_size=self.kernel_size,
                    device=self.device
                ).to(self.device)

                kernels.append(blur_kernel.get_kernel().unsqueeze(0).float())
            return torch.stack(kernels).to(self.device)

    def get_latent_kernel(self, total_number: int) -> torch.Tensor:
        kernel_np = np.random.randn(total_number, 512, 2, 2) * self.intensity
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        return random_kernel
