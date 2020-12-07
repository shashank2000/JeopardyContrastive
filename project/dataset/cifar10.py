from torchvision import datasets
import torch.utils.data as data
class CIFAR10Modified(data.Dataset):
    def __init__(self, train=True, image_transforms=None):
        super().__init__()
        self.dataset = datasets.CIFAR10(
            root="/data5/wumike/cifar10",
            train=train,
            transform=image_transforms
        )

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        return index, img_data, img2_data, label

    def __len__(self):
        return len(self.dataset)
        
class CIFAR100Modified(data.Dataset):
    def __init__(self, train=True, image_transforms=None):
        super().__init__()
        self.dataset = datasets.CIFAR100(
            root="/data5/jasmine7/cifar100",
            train=train,
            transform=image_transforms
        )

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        return index, img_data, img2_data, label

    def __len__(self):
        return len(self.dataset)