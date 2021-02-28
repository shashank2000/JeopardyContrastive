from torch.nn import Linear, Tanh, LSTM, Embedding, Sequential, ReLU, BatchNorm1d
from torch.optim import SGD, Adam
# from pytorch_lightning import LightningModule, EvalResult, TrainResult
import pytorch_lightning as pl
from torchvision.models import resnet18
from loss_func.simclr import SimCLR
import torch
from torchvision import transforms 
from model_utils import pretrain_optimizer, Projection, get_pretrained_emb_layer, pretrain_scheduler
import os
import numpy as np
import math
from pytorch_lightning.utilities import AMPType
from torch.optim.optimizer import Optimizer

class v3ModelCross(pl.LightningModule):
    def __init__(self, vocab_sz, config, num_samples=1000):
      '''
      Trying symmetric loss
      (image+question) -> answer (negatives)
      (image+answer) -> question(negatives)
      (question+answer) -> image(negatives)

      '''
      super().__init__()
      self.save_hyperparameters()
      mp = config.model_params
      self.op = config.optim_params
      self.im_vec_dim = mp.im_vec_dim
      self.ans_dim = mp.ans_dim
      self.question_dim = mp.question_dim
      self.n_hidden = mp.n_hidden
      self.n_layers = mp.n_layers # in case we want multilayer RNN
      self.tau = mp.tau
      # initialize with Glove embeddings to have accuracy skyrocket
      emb_layer = get_pretrained_emb_layer()
      self.i_h = Embedding.from_pretrained(emb_layer, padding_idx=400000-1)

      self.h_o = Sequential(
        Linear(2*self.n_hidden, self.question_dim),
        ReLU(),
        Linear(self.question_dim, self.question_dim)
      )
      
      self.ans_final = Linear(self.n_hidden, self.question_dim)
      
      if self.n_layers > 1:
        self.rnn = LSTM(self.n_hidden, self.n_hidden, self.n_layers, dropout=0.5, batch_first=False)
      else:
        self.rnn = LSTM(self.n_hidden, self.n_hidden)

      self.image_feature_extractor = resnet18(pretrained=False)
      self.image_feature_extractor.fc = Linear(512, self.question_dim)
      
      self.projection_head = Projection(self.question_dim, mp.proj_hidden, mp.proj_output)
      
      # compute iters per epoch
      train_iters_per_epoch = num_samples // self.op.batch_size
      self.lr_schedule = pretrain_scheduler(
        self.op.learning_rate, train_iters_per_epoch, 
        config.num_epochs, config.scheduler_params
      )

    def forward(self, x):
      return self.image_feature_extractor(x)

    def forward_question(self, x):
      # we have a question as input, and take the final hidden state as output
      _, (hn, cn) = self.rnn(self.i_h(x)) # hidden state has 50 features, is this enough?
      # concatenate and transpose
      res = torch.cat((hn.squeeze(), cn.squeeze()), dim=1) # hn and cn are now both shape 256, 50
      res = self.h_o(res)
      return res

    def forward_answer(self, x):
      x = self.i_h(x)
      return self.ans_final(x)
        
    def training_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      self.log("train_loss", loss, prog_bar=True)
      return loss 

    def shared_step(self, batch):
      question, image, answer = batch
      question = torch.stack(question) # becomes (10, 256) vector
      
      f_q = self.forward_question(question)
    
      f_a = self.forward_answer(answer)
      
      f_im = self(image)

      # (image+question) -> answer (negatives)
      im_q_vector = f_im * f_q
      loss = SimCLR(im_q_vector, f_a, self.tau).get_loss()

      # (image+answer) -> question(negatives)
      im_a_vector = f_im * f_a
      loss += SimCLR(im_a_vector, f_q, self.tau).get_loss()

      # (question+answer) -> image(negatives)
      q_a_vector = f_q * f_a
      loss += SimCLR(q_a_vector, f_im, self.tau).get_loss()

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