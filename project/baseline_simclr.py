from torch.nn import Linear, Tanh
from torch.optim import SGD
import pytorch_lightning as pl
from torchvision.models import resnet18
from loss_func.simclr import SimCLR
import torch

class UpperBoundModel(pl.LightningModule):
    def __init__(self, config):
      # possible next step - use auto scaling of batch size on GPU
      super().__init__()
      mp = config.model_params
      self.op = config.optim_params
      self.im_vec_dim = mp.im_vec_dim
      
      # for images, we use resnet18, and modify the number of output classes
      self.image_feature_extractor = resnet18(pretrained=False)
      self.image_feature_extractor.fc = Linear(512, self.im_vec_dim)
      
    def forward(self, x):
      return self.image_feature_extractor(x)
 
    def training_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      self.log("train_loss", loss, prog_bar=True)
      return loss 

    def validation_step(self, batch, batch_idx):
      # what hyperparams am I varying?
      loss = self.shared_step(batch)
      self.log("val_loss", loss, prog_bar=True)
      return loss 

    def shared_step(self, batch):
      # we are only pretraining, the transfer task is where we care about labels
      _, im1, im2, _ = batch
      im1v = self(im1)
      im2v = self(im2)
      loss = SimCLR(im1v, im2v).get_loss() # trying to maximize the similarity between these two images
      return loss

    def configure_optimizers(self):
      return SGD(self.parameters(), momentum=self.op.momentum, weight_decay=self.op.weight_decay, lr=self.op.learning_rate)
