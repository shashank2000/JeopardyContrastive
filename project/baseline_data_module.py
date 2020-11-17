from torchvision import transforms
from dataset.mscoco import MSCOCO
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch import tensor

class BaselineDataModule(LightningDataModule):
  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size
    
    self.test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # random crop, color jitter etc 
    self.train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur([1, 1])], p=0.5), # perhaps this blur is too much
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                    std=[0.247, 0.243, 0.261]),
            ])

    self.train_dataset = MSCOCO(train=True, image_transforms=self.train_transform)
    self.test_dataset = MSCOCO(train=False, image_transforms=self.test_transform)
    
  def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size,
                        num_workers=48, pin_memory=True, drop_last=True)

  def val_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.batch_size,
                        num_workers=48, pin_memory=True, drop_last=True)