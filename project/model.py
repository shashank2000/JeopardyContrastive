from torch.nn import Linear, Tanh, LSTM, Embedding
from torch.optim import SGD
import pytorch_lightning as pl 
from torchvision.models import resnet18
from loss_func.simclr import SimCLR
import torch

class JeopardyModel(pl.LightningModule):
    def __init__(self, vocab_sz, bs):
      super().__init__()
      im_vec_dim = 128
      ans_dim = 128
      question_dim = 256
      n_hidden = 100
      n_layers = 1
      
      self.i_h = Embedding(vocab_sz, n_hidden)
      self.rnn = LSTM(n_hidden, n_hidden)
      self.h_o = Linear(n_hidden, question_dim)

      self.h = None 
      self.ans_final = Linear(n_hidden, ans_dim)

      # for images, we use resnet18 pretrained
      self.image_feature_extractor = resnet18(pretrained=True, progress=True)
      for p in self.image_feature_extractor.parameters():
        p.requires_grad = False

      self.image_feature_extractor.fc = Linear(512, im_vec_dim)
      
    def forward(self, x):
      return self.image_feature_extractor(x)

    def forward_question(self, x):
      # we have a question as input
      if not self.h:
        self.h = torch.zeros(1, 16, 100, device=self.device), torch.zeros(1, 16, 100, device=self.device) # h0, c0
      res, h = self.rnn(self.i_h(x), self.h)
      self.h = h[0].detach(), h[1].detach() # don't want to backpropagate all the way through the RNN
      return self.h_o(res)

    def forward_answer(self, x):
      # just a linear layer over the embeddings to begin with
      x = self.i_h(x)
      return self.ans_final(x)
        

    def training_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      result = pl.TrainResult(loss)
      result.log_dict({'train_loss': loss}, prog_bar=True)
      return result

    def validation_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
      result.log_dict({'val_loss': loss}, prog_bar=True)
      return result

    def test_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      result = pl.EvalResult()
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
      return SGD(self.parameters(), lr=2e-2)
