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
                transforms.RandomApply([transforms.GaussianBlur([1, 1])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                    std=[0.247, 0.243, 0.261]),
            ])

    # train and test transforms needed
    self.questions_file = "/data5/shashank2000/final_json/OpenEnded_mscoco_train2014_questions.json"
    self.answers_file = "/data5/shashank2000/final_json/mscoco_train2014_annotations.json"
    self.coco_loc = "/mnt/fs0/datasets/mscoco/train2014"    
    # Enter path for questions_file, answers_file and directory for COCO images that correspond to these
    #self.questions_file = "../../datasets/questions.json"
    #self.answers_file = "../../datasets/answers.json"

    # copy and then unarchive instead?
    #self.coco_loc = "../../datasets/train2014"
    # do something about self.dims?
    self.train_dataset = JeopardyDataset(self.questions_file, self.answers_file, self.coco_loc, self.train_transform, train=True)
    self.test_dataset = JeopardyDataset(self.questions_file, self.answers_file, self.coco_loc, self.test_transform, 
        word2idx=self.train_dataset.word2idx, train=False)
    
    self.vl = self.get_vocab_length()
    
  def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size,
                        num_workers=48, pin_memory=True, drop_last=True)

  def val_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.batch_size,
                        num_workers=48, pin_memory=True, drop_last=True)

  def get_vocab_length(self):
    return self.train_dataset.vocabulary_length()
