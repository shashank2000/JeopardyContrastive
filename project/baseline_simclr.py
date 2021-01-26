from torch.optim.optimizer import Optimizer
import torch.nn as nn
from pytorch_lightning.utilities import AMPType
from torch.optim import SGD
import pytorch_lightning as pl
from torchvision.models import resnet18
from loss_func.simclr import SimCLR
import torch
from utils.model_utils import pretrain_optimizer, Projection, pretrain_scheduler
import numpy as np
import math

class UpperBoundModel(pl.LightningModule):
    def __init__(self, config, num_samples=1000):
      # possible next step - use auto scaling of batch size on GPU
      super().__init__()
      mp = config.model_params
      self.op = config.optim_params

      # for images, we use resnet18, and modify the number of output classes
      self.image_feature_extractor = resnet18(pretrained=False)

      self.image_feature_extractor.fc = nn.Sequential(
        nn.Linear(512, mp.image_size),     
        Projection(input_dim=mp.image_size, hidden_dim=mp.proj_hidden, output_dim=mp.proj_output)
      )

      # compute iters per epoch
      train_iters_per_epoch = num_samples // self.op.batch_size
      breakpoint()
      self.lr_schedule = pretrain_scheduler(
        self.op.learning_rate, train_iters_per_epoch, 
        config.num_epochs, config.scheduler_params
      )
      
    def forward(self, x):
      return self.image_feature_extractor(x)
 
    def training_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      self.log("train_loss", loss, prog_bar=True)
      return loss 

    def shared_step(self, batch):
      # we are only pretraining, the transfer task is where we care about labels
      _, im1, im2, _ = batch
      im1v = self(im1)
      im2v = self(im2)
      loss = SimCLR(im1v, im2v).get_loss() # trying to maximize the similarity between these two images
      return loss
    
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        optimizer_closure = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        for param_group in optimizer.optim.param_groups:
          param_group["lr"] = self.lr_schedule[self.trainer.global_step]
        
        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        self.log('learning_rate', self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        # from lightning
        if self.trainer.amp_backend == AMPType.NATIVE:
            optimizer_closure()
            self.trainer.scaler.step(optimizer)
        elif self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
      # TODO: add exclude_bn_bias flag
      return pretrain_optimizer(self.parameters(), self.op.momentum, self.op.weight_decay, self.op.learning_rate, lars=True)