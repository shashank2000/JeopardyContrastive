from torchvision import transforms
from dataset.jeopardy_dataset import JeopardyDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch import tensor

class VQADataModule(LightningDataModule):
  def __init__(self, batch_size, val_split=0.2):
    super().__init__()
    self.batch_size = batch_size
    self.val_split = val_split
    self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Enter path for questions_file, answers_file and directory for COCO images that correspond to these
    self.questions_file = "../../datasets/questions.json"
    self.answers_file = "../../datasets/answers.json"

    # copy and then unarchive instead?
    self.coco_loc = "../../datasets/train2014"
    # do something about self.dims?
    self.dataset = JeopardyDataset(self.questions_file, self.answers_file, self.coco_loc, self.transform)

    self.vl = self.get_vocab_length()

  def setup(self, stage=None):  
    dataset_size = len(self.dataset)
    indices = list(range(dataset_size))
    split = self.dataset.get_split_index()
    train_indices, val_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    self.train_sampler = SubsetRandomSampler(train_indices)
    self.val_sampler = SubsetRandomSampler(val_indices)
    
  def train_dataloader(self):
      return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler,
                        num_workers=4, pin_memory=True, drop_last=True)

  def val_dataloader(self):
      return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.val_sampler,
                        num_workers=4, pin_memory=True, drop_last=True)

  def get_vocab_length(self):
    return self.dataset.vocabulary_length()