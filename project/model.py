from torch.nn import Linear, Tanh, LSTM, Embedding
from torch.optim import SGD, Adam
from pytorch_lightning import LightningModule, EvalResult, TrainResult
from torchvision.models import resnet18
from loss_func.simclr import SimCLR
import torch
from torchvision import transforms 


class JeopardyModel(LightningModule):
    def __init__(self, vocab_sz, im_vec_dim=128, ans_dim=128, question_dim=256, n_hidden=100, n_layers=1):
      # possible next step - use auto scaling of batch size on GPU
      super().__init__()
      self.im_vec_dim = im_vec_dim
      self.ans_dim = ans_dim
      self.question_dim = question_dim
      self.n_hidden = n_hidden
      self.n_layers = n_layers # in case we want multilayer RNN
      
      self.i_h = Embedding(vocab_sz, n_hidden, padding_idx=0)  
      self.h_o = Linear(n_hidden, question_dim)
      self.h = None 
      self.ans_final = Linear(n_hidden, ans_dim)
      if self.n_layers > 1:
        self.rnn = LSTM(n_hidden, n_hidden, n_layers)
      else:
        self.rnn = LSTM(n_hidden, n_hidden)

      # for images, we use resnet18, and modify the number of output classes
      self.image_feature_extractor = resnet18(pretrained=False)
      self.image_feature_extractor.fc = Linear(512, im_vec_dim)
      
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
      result = TrainResult(loss)
      result.log_dict({'train_loss': loss}, prog_bar=True)
      return result

    def validation_step(self, batch, batch_idx):
      # what hyperparams am I varying?
      loss = self.shared_step(batch)
      result = EvalResult(checkpoint_on=loss, early_stop_on=loss)
      result.log_dict({'val_loss': loss}, prog_bar=True)
      return result

    def test_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      result = EvalResult()
      result.log_dict({'test_loss': loss}, prog_bar=True)
      return result

    def shared_step(self, batch):
      question, image, answer = batch
      question = torch.stack(question)
      f_q = self.forward_question(question)
      f_q = f_q.squeeze()[-1, :]
      f_a = self.forward_answer(answer)
      im_vector = self(image)
      answer_image_vector = torch.cat((f_a, im_vector), 1)
      loss = SimCLR(answer_image_vector, f_q).get_loss()
      return loss

    def configure_optimizers(self):
      return Adam(self.parameters(), lr=2e-2)
