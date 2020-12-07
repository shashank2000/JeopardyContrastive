from torch.nn import Linear, Tanh, LSTM, Embedding
from torch.optim import SGD, Adam
# from pytorch_lightning import LightningModule, EvalResult, TrainResult
import pytorch_lightning as pl
from torchvision.models import resnet18
from loss_func.simclr import SimCLR
import torch
from torchvision import transforms 

class JeopardyModel2(pl.LightningModule):
    def __init__(self, vocab_sz, config):
      # possible next step - use auto scaling of batch size on GPU
      super().__init__()
      mp = config.model_params
      self.op = config.optim_params
      self.im_vec_dim = mp.im_vec_dim
      self.ans_dim = mp.ans_dim
      self.question_dim = mp.question_dim
      self.n_hidden = mp.n_hidden
      self.n_layers = mp.n_layers # in case we want multilayer RNN

      self.i_h = Embedding(vocab_sz, self.n_hidden, padding_idx=0)  
      self.h_o = Linear(self.n_hidden, self.question_dim)
      self.h = None 
      self.ans_final = Linear(self.n_hidden, self.ans_dim)
      if self.n_layers > 1:
        self.rnn = LSTM(self.n_hidden, self.n_hidden, self.n_layers)
      else:
        self.rnn = LSTM(self.n_hidden, self.n_hidden)

      # for images, we use resnet18, and modify the number of output classes
      self.image_feature_extractor = resnet18(pretrained=False)
      self.image_feature_extractor.fc = Linear(512, self.im_vec_dim)
      
    def forward(self, x):
      return self.image_feature_extractor(x)

    def forward_question(self, x):
      # we have a question as input
      if not self.h:
        batch_size = x.shape[1]
        self.h = torch.zeros(1, batch_size, self.n_hidden, device=self.device), torch.zeros(1, batch_size, self.n_hidden, device=self.device) # h0, c0
      res, h = self.rnn(self.i_h(x), self.h)
      self.h = h[0].detach(), h[1].detach()
      return self.h_o(res)

    def forward_answer(self, x):
      # just a linear layer over the embeddings to begin with
      x = self.i_h(x)
      return self.ans_final(x)
        
    def training_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      self.log("train_loss", loss, prog_bar=True)
      return loss 
      
    def shared_step(self, batch):
      # we test a question-image vector instead - so not quite Jeopardy
      question, image, answer = batch
      question = torch.stack(question)
      f_q = self.forward_question(question)
      f_q = f_q.squeeze()[-1, :]
      f_a = self.forward_answer(answer)
      im_vector = self(image)
      question_image_vector = torch.cat((f_q, im_vector), 1)
      loss = SimCLR(question_image_vector, f_a).get_loss()
      return loss

    def configure_optimizers(self):
      return SGD(self.parameters(), momentum=self.op.momentum, weight_decay=self.op.weight_decay, lr=self.op.learning_rate)
