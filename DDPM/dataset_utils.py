import os.path
import sys

# sys.path.append('.')

import torch
import torchvision.datasets
import numpy as np

# from measures import ssim


class MIACIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, idxs, **kwargs):
        super(MIACIFAR10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR10, self).__getitem__(item)


class MIAImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self, idxs, **kwargs):
        super(MIAImageFolder, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIAImageFolder, self).__getitem__(item)


def load_member_data(dataset_name, batch_size=128, shuffle=False, randaugment=False):
    member_split_root = ''
    dataset_root = 'data/datasets/pytorch'
    if dataset_name.upper() == 'CIFAR10':
        splits = np.load(os.path.join(member_split_root, 'CIFAR10_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        if randaugment:
            transforms = torchvision.transforms.Compose([torchvision.transforms.RandAugment(num_ops=5),
                                                         torchvision.transforms.ToTensor()])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
        member_set = MIACIFAR10(member_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                transform=transforms)
        nonmember_set = MIACIFAR10(nonmember_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                   transform=transforms)
    elif dataset_name.upper() == 'TINY-IN':
        splits = np.load(os.path.join(member_split_root, 'TINY-IN_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIAImageFolder(member_idxs, root=os.path.join("data/datasets", 'tiny-imagenet-200/train'),
                                    transform=transforms)
        nonmember_set = MIAImageFolder(nonmember_idxs, root=os.path.join("data/datasets", 'tiny-imagenet-200/train'),
                                       transform=transforms)
    else:
        raise NotImplementedError

    member_loader = torch.utils.data.DataLoader(member_set, batch_size=batch_size, shuffle=shuffle)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_set, batch_size=batch_size, shuffle=shuffle)
    return member_set, nonmember_set, member_loader, nonmember_loader
