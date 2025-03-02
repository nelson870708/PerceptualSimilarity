################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################

import os
import os.path

import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]

NP_EXTENSIONS = [
    ".npy",
]


def is_right_file(filename, mode="img"):
    if mode == "img":
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif mode == "np":
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)


def make_dataset(dirs, mode="img"):
    if not isinstance(dirs, list):
        dirs = [
            dirs,
        ]

    paths = []
    for dir in dirs:
        assert os.path.isdir(dir), "%s is not a valid directory" % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_right_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    paths.append(path)

    # print("Found %i images in %s"%(len(images),root))
    return paths


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in: " + root + "\n"
                                                   "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
