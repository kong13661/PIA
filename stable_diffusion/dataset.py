from typing import Any

import torch
from torchvision.datasets import CocoDetection as _CocoDetection, VisionDataset
import os
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
import numpy as np
import pandas as pd

coco_dataset_root = ''
coco_dataset_anno = ''
stable_diffusion_data = ''


class ImageOnlyDataset(VisionDataset):
    def __init__(self, root: str, transforms = None, transform = None, target_transform = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.files = os.listdir(root)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Any:
        img = Image.open(os.path.join(self.root, self.files[index])).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.files[index]

    @staticmethod
    def collate_fn(examples):
        imgs = []
        captions = []
        for e in examples:
            imgs.append(e[0])
            captions.append(e[1])
        return torch.stack(imgs), captions

    def __len__(self) -> int:
        return len(self.ids)


class Laion5(VisionDataset):
    def __init__(self, root: str, metadata, transforms = None, transform = None, target_transform = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.metadata = np.load(metadata)

    def __getitem__(self, index: int) -> Any:
        img = Image.open(os.path.join(self.root, self.metadata[index][0] + ".jpg")).convert('RGB')
        caption = self.metadata[index][1]
        if self.transforms is not None:
            img, caption = self.transforms(img, caption)
        return img, caption

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def collate_fn(examples):
        imgs = []
        captions = []
        for e in examples:
            imgs.append(e[0])
            captions.append(e[1])
        return torch.stack(imgs), captions


class Laion5Generated(Laion5):
    def __init__(self, root: str, metadata, generated, transforms = None, transform = None, target_transform = None) -> None:
        super().__init__(root, metadata, transforms, transform, target_transform)
        self.generated = pd.read_csv(generated)

    def __getitem__(self, index: int) -> Any:
        img = Image.open(os.path.join(self.root, self.metadata[index][0] + ".jpg")).convert('RGB')
        caption = self.generated[self.generated.file_name == self.metadata[index][0] + ".jpg"].iloc[0, 0]
        if self.transforms is not None:
            img, caption = self.transforms(img, caption)
        return img, caption


class CocoDetection(_CocoDetection):
    def __init__(self, root: str, annFile: str, transform = None, target_transform = None, transforms = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.splited_id = OmegaConf.load(f'coco-2500-random.yaml')

    def __len__(self) -> int:
        return len(self.splited_id)

    def __getitem__(self, index: int):
        return super().__getitem__(self.splited_id[index])

    @staticmethod
    def collate_fn(examples):
        imgs = []
        captions = []
        for e in examples:
            imgs.append(e[0])
            captions.append(e[1][0]['caption'])
        return torch.stack(imgs), captions


class CocoDetectionGenerated(CocoDetection):
    def __init__(self, root: str, annFile: str, generated, transform = None, target_transform = None, transforms = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.generated = pd.read_csv(generated)

    def __getitem__(self, index: int):
        img, _ = super().__getitem__(index)
        file_name = self.coco.loadImgs(self.ids[self.splited_id[index]])[0]["file_name"]
        caption = self.generated[self.generated.file_name == file_name].iloc[0, 0]
        return img, caption

    @staticmethod
    def collate_fn(examples):
        imgs = []
        captions = []
        for e in examples:
            imgs.append(e[0])
            captions.append(e[1])
        return torch.stack(imgs), captions


class CocoImageOnlyDataset(CocoDetection):
    def __getitem__(self, index: int):
        return super().__getitem__(index)[0], \
            self.coco.loadImgs(self.ids[self.splited_id[index]])[0]["file_name"]


def load_member_data(dataset_name, batch_size=8):
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    if dataset_name == 'laion5':
        member_set = Laion5(f"{stable_diffusion_data}/images-random",
                            f"{stable_diffusion_data}/val-list-2500-random.npy",
                            transform=transform)
        nonmember_set = CocoDetection(root=coco_dataset_root,
                                      annFile=coco_dataset_anno,
                                      transform=transform)
    elif dataset_name == 'laion5_none':
        member_set = Laion5(f"{stable_diffusion_data}/images-random",
                            f"{stable_diffusion_data}/val-list-2500-random.npy",
                            transform=transform)
        nonmember_set = CocoDetection(root=coco_dataset_root,
                                      annFile=coco_dataset_anno,
                                      transform=transform)
    elif dataset_name == 'laion5_blip':
        member_set = Laion5Generated(f"{stable_diffusion_data}/images-random",
                                     f"{stable_diffusion_data}/val-list-2500-random.npy",
                                     f'{stable_diffusion_data}/text_generation/images-random.csv',
                                     transform=transform)
        nonmember_set = CocoDetectionGenerated(root=coco_dataset_root,
                                               annFile=coco_dataset_anno,
                                               generated=f'{stable_diffusion_data}/text_generation/val2017.csv',
                                               transform=transform)

    member_loader = torch.utils.data.DataLoader(member_set, batch_size=batch_size, collate_fn=member_set.collate_fn)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_set, batch_size=batch_size, collate_fn=nonmember_set.collate_fn)
    return member_set, nonmember_set, member_loader, nonmember_loader


if __name__ == '__main__':
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
