from pytorch_lightning import Trainer, seed_everything
from dataset.mscoco import BaseMSCOCO
import pytorch_lightning as pl
import torch.nn as nn
from model_im_q_a import JeopardyModel2
from model import JeopardyModel
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch
from pytorch_lightning.utilities import AMPType

from pytorch_lightning.utilities import AMPType
from torch.optim.optimizer import Optimizer

class DumbJeopardyTest(pl.LightningModule):
    def __init__(self, main_model_path, parent_config, config, vocab_sz=None, num_classes=10000):
        super().__init__()
        self.save_hyperparameters()
        breakpoint()
        if parent_config.system == "inverse-jeopardy":
            self.main_model = JeopardyModel2.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
        else:
            self.main_model = JeopardyModel.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
            
        self.op = config.optim_params
        self.main_model.freeze()
        self.q_rnn = self.main_model.rnn
        self.q_embed = self.main_model.i_h # learned embedding layer, was initialized to Glove Embeddings in pretrained task
        self.h = self.main_model.h

        self.resnet = self.main_model.image_feature_extractor

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.test_accuracy =  pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy() # to check for overfitting!
        self.top_k_accuracy = pl.metrics.Accuracy()
        self.op = config.optim_params
        self.mp = config.model_params
        
        # input_dim = self.mp.q_dim + self.mp.im_dim # 512, which happens to be same as post-pool dimension
        input_dim = self.mp.q_dim + 512
        self.n_hidden = self.mp.n_hidden # should be same as number of hidden layers in pretraining network
        self.h_o = self.main_model.h_o # outputs a 256 dim vector
        # self.fine_tune_image = nn.Linear(512, self.mp.im_dim)
        
        self.fine_tune = nn.Linear(input_dim, num_classes)
            
        self.h = None
        
    def forward(self, x):
        x = self.fine_tune(x)
        return nn.functional.log_softmax(x, dim=1)

    def forward_image(self, x):
        x = self.resnet(x)
        x = x.view(self.op.batch_size, -1)
        return x

    def forward_question(self, x):
        res, h = self.q_rnn(self.q_embed(x), self.h) 
        self.h = h[0].detach(), h[1].detach()
        return self.h_o(res)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, testing=True)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def shared_step(self, batch, testing=False):
        question, image, answer = batch
        question = torch.stack(question)
        f_q = self.forward_question(question) # its 10, 256, 256
        f_q = f_q.squeeze()[-1, :] # 256
        im_vector = self.forward_image(image)
        im_vector = im_vector.squeeze()

        # TODO: "element-wise product was FAR superior to a concatenation"
        question_image_vector = torch.cat((f_q, im_vector), 1)
        # question_image_vector = f_q * im_vector
        # pass this through batchnorm, fcc, nonlinearity??
        logits = self(question_image_vector)
        # loss is just cross-entropy loss between answer and question_image vector   
        loss = nn.functional.nll_loss(logits, answer)
        if not testing:
            acc = self.train_accuracy(logits, answer)
            self.log('train_acc', acc)
        else:
            acc = self.test_accuracy(logits, answer)
            top_k_accuracy = None
            vec = torch.topk(logits, 5).indices
            for i, a in enumerate(answer):
                found = False
                for v in vec[i]:
                    if v == a:
                        top_k_accuracy = self.top_k_accuracy(a, a)
                        found = True
                        break
                if not found:
                    top_k_accuracy = self.top_k_accuracy(a+1,a)
            self.log('top_k_accuracy', top_k_accuracy)
            self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        return SGD(self.parameters(), self.op.momentum, self.op.weight_decay, self.op.learning_rate)