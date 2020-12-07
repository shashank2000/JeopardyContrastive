from torch import tensor
from PIL import ImageFilter, Image
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

from dataset.mscoco import MSCOCO
# random crop, color jitter etc 
train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1,2.])], p=0.5), # perhaps this blur is too much
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446],
                            std=[0.247, 0.243, 0.261]),
    ])

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446],
                            std=[0.247, 0.243, 0.261]),
    ])



def get_dataloaders(config, train_trans=train_transform, test_trans=test_transform, val=False):
    # config contains batch size, dataset, num_workers, split is always 0.2
    train_dataset = None
    test_dataset = None
    if ds == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="/data5/wumike/cifar10",
            train=True,
            transform=train_trans
        )
        test_dataset = datasets.CIFAR10(
            root="/data5/wumike/cifar10",
            train=False,
            transform=test_trans
        )
    else:
        train_dataset = BaseMSCOCO(image_transforms=image_transforms)
        test_dataset = BaseMSCOCO(train=False, image_transforms=image_transforms)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(0.2*dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=bs, num_workers=48)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, num_workers=48)
    if not test:
        return train_dataloader, test_dataloader

    val_dataloader = DataLoader(train_dataset, sampler=val_sampler, batch_size=bs, num_workers=48)

    return train_dataloader, val_dataloader, test_dataloader

