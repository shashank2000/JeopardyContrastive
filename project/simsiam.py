'''
The implementation for SimSiam jeopardy - comparing an image-answer 
and question view. Image-answer is made by taking the element-wise product
of the image vector and answer vector. 

SGD with a learning rate of lr * (batch_size)/256 with a base lr = 0.05,
The learning rate has a cosine decay schedule [27, 8].
The weight decay is 0.0001 and the SGD momentum is 0.9.
'''
from torch.nn import Linear, Tanh, LSTM, Embedding, Sequential, ReLU, BatchNorm1d
from torch.optim import SGD, Adam
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
from utils.simsiam import SimSiamSystem

class SimSiamJeopardy(pl.LightningModule):
    def __init__(self, config):
      super().__init__()
      self.save_hyperparameters()
      mp = config.model_params
      self.op = config.optim_params
      self.im_vec_dim = mp.im_vec_dim
      self.ans_dim = mp.ans_dim
      self.question_dim = mp.question_dim
      self.n_hidden = mp.n_hidden
      self.n_layers = mp.n_layers # in case we want multilayer RNN
      
      emb_layer = get_pretrained_emb_layer()
      self.i_h = Embedding.from_pretrained(emb_layer, padding_idx=400000-1)
      
      if self.n_layers > 1:
        self.rnn = LSTM(self.n_hidden, self.n_hidden, self.n_layers, dropout=0.5, batch_first=False)
      else:
        self.rnn = LSTM(self.n_hidden, self.n_hidden)

      self.image_feature_extractor = resnet18(pretrained=False)
      self.image_feature_extractor.fc = Linear(512, self.question_dim)
      
      self.ans_final = Linear(self.n_hidden, self.ans_dim)
      self.encoder = self.get_encoder()
      self.simsiam_system = SimSiamSystem(config, self.encoder) # we might want to play with different combinations of objects

    def get_encoder(self, images=None, questions=None, answers=None):
        # the encoder gets as input either answers, questions or images or combination
        # core encoder logic can change as needed
        if images and answers:
            # pass images through resnet, answers through linear layer
            # take element-wise product at the end of the day
            im_new = self.forward(images)
            a_new = self.forward_answer(answers)
            return im_new * a_new
        elif questions:
            q_new = self.forward_question(questions)
            return q_new
        
    def forward(self, x):
      return self.image_feature_extractor(x)

    def forward_question(self, x):
      # we have a question as input, and take the final hidden state as output
      _, (hn, cn) = self.rnn(self.i_h(x)) # hidden state has 50 features, is this enough?
      # concatenate and transpose
      res = torch.cat((hn.squeeze(), cn.squeeze()), dim=1) # hn and cn are now both shape 256, 50
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
      loss = self.simsiam_system.get_loss(
          images=image, 
          answers=answer, 
          questions=question)
      return loss

    def configure_optimizers(self):
      return pretrain_optimizer(self.parameters(), self.op.momentum, self.op.weight_decay, self.op.learning_rate, lars=True)

