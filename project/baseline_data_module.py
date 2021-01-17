from torchvision import transforms, datasets
from dataset.mscoco import MSCOCO, BaseMSCOCO
from dataset.cifar10 import CIFAR10Modified, CIFAR100Modified
from dataset.imagenet import ImageNet
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch import tensor
import random
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class BaselineDataModule(LightningDataModule):
  def __init__(self, batch_size, num_workers, dataset_type, train_transform=None, test_transform=None):
    super().__init__()
    self.batch_size = batch_size
    
    self.test_transform = test_transform or transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446],
                            std=[0.247, 0.243, 0.261]),
    ])

    self.num_workers = num_workers

    # random crop, color jitter etc 
    self.train_transform = train_transform or transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5), # perhaps this blur is too much
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                    std=[0.247, 0.243, 0.261]),
            ])

    # for both linear eval and fine tuning
    self.fine_tune_train_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.491, 0.482, 0.446],
                          std=[0.247, 0.243, 0.261])
    ])

    self.train_dataset, self.test_dataset = self.get_datasets(dataset_type)
    print(len(self.train_dataset))
    
  def get_datasets(self, dataset_type):
    train_dataset, test_dataset = None, None

    # no test dataset for contrastive pretraining tasks
    if dataset_type == "coco-contrastive":
      train_dataset = MSCOCO(train=True, image_transforms=self.train_transform)
    elif dataset_type == "cifar-contrastive":
      train_dataset = CIFAR10Modified(train=True, image_transforms=self.train_transform)
    elif dataset_type == "cifar100-contrastive":
      train_dataset = CIFAR100Modified(train=True, image_transforms=self.train_transform)
    elif dataset_type == "cifar10":
      train_dataset = datasets.CIFAR10(
            root="/data5/wumike/cifar10",
            train=True,
            transform=self.fine_tune_train_transform
        )
      test_dataset = datasets.CIFAR10(
            root="/data5/wumike/cifar10",
            train=False,
            transform=self.test_transform
        )
    elif dataset_type == "coco":
      train_dataset = BaseMSCOCO(image_transforms=self.fine_tune_train_transform)
      test_dataset = BaseMSCOCO(train=False, image_transforms=self.test_transform)
    elif dataset_type == "cifar100":
      train_dataset = datasets.CIFAR100(
            root="/data5/jasmine7/cifar100",
            train=True,
            transform=self.fine_tune_train_transform
        )
      test_dataset = datasets.CIFAR100(
            root="/data5/jasmine7/cifar100",
            train=False,
            transform=self.test_transform
        )
    elif dataset_type == "stl10":
      train_dataset = datasets.STL10(
            root="/data5/shashank2000/stl10",
            split='train',
            transform=self.fine_tune_train_transform,
            download=True
        )
      test_dataset = datasets.STL10(
            root="/data5/shashank2000/stl10",
            split='test',
            transform=self.test_transform,
            download=True
        )
    elif dataset_type == "imagenet":
      train_dataset = ImageNet(train=True, image_transforms=self.fine_tune_train_transform)
      test_dataset = ImageNet(train=False, image_transforms=self.test_transform)
    
    return train_dataset, test_dataset
      
  def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size,
                        num_workers=self.num_workers, pin_memory=True, drop_last=True)
  
  def test_dataloader(self):
      # no test step defined in pretraining, this dataloader is only used by the finetune network
      return DataLoader(self.test_dataset, batch_size=self.batch_size,
                        num_workers=self.num_workers, pin_memory=True, drop_last=True)