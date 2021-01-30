from pytorch_lightning import Trainer, seed_everything
from dataset.mscoco import BaseMSCOCO
import pytorch_lightning as pl
import torch.nn as nn
from baseline_simclr import UpperBoundModel
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import torch
from pytorch_lightning.utilities import AMPType
from utils.model_utils import get_pretrained_emb_layer
from pytorch_lightning.utilities import AMPType
from torch.optim.optimizer import Optimizer
from torch.nn import Embedding

def print_sentence(indices, word_index_to_word):
    for i in indices:
        print(word_index_to_word[i.item()])


class BaselineVQA(pl.LightningModule):
    def __init__(self, main_model_path, parent_config, config, word_index_to_word, num_samples=1000, num_classes=10000):
        '''
            The idea with the VQA baseline is that we take the SimCLR ResNet, and take the average Glove embedding for the question,
            and finally predict the answer using those two vectors (concatenated).

            Uses same SGD as vqatransfer, only difference here is we take the pretrained ResNet from Upperbound, 
            and weights for questions are just average Glove embeddings. 
        '''
        super().__init__()
        # self.save_hyperparameters([main_model_path, parent_config, config])
        self.word_index_to_word = word_index_to_word
        # we don't really care about num_samples, just needs to be the same as the pretrained model for checkpoint consistency
        self.main_model = UpperBoundModel.load_from_checkpoint(main_model_path, config=parent_config)
        self.main_model.freeze()
    
        self.op = config.optim_params
        self.mp = config.model_params

        emb_layer = get_pretrained_emb_layer()
        self.i_h = Embedding.from_pretrained(emb_layer, padding_idx=400000-1) 

        self.resnet = self.main_model.image_feature_extractor
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.test_accuracy =  pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy() # to check for overfitting!
        self.top_k_accuracy = pl.metrics.Accuracy()
        
        input_dim = self.mp.q_dim + self.mp.im_dim # 512, which happens to be same as post-pool dimension
        
        self.n_hidden = self.mp.n_hidden # should be same as number of hidden layers in pretraining network
        self.fine_tune_questions = nn.Linear(self.mp.glove_dim, self.mp.q_dim) # i_h's output -> mp.q_dim
        self.fine_tune_image = nn.Linear(512, self.mp.im_dim) # resnet output to 256d
        
        self.fine_tune = nn.Linear(input_dim, config.answer_classes + 1)
            
    def forward(self, x):
        return self.fine_tune(x)

    def forward_image(self, x):
        x = self.resnet(x)
        x = x.view(self.op.batch_size, -1)
        return self.fine_tune_image(x)

    def forward_question(self, x):
        # average the glove embeddings, and then run through h_o
        mask = (x != 400000-1).long() # padding index
        mask1 = mask.unsqueeze(2).repeat(1, 1, self.mp.glove_dim)
        x = self.i_h(x)
        res = torch.sum(x*mask1, dim=1)/torch.sum(mask, dim=1, keepdim=True)
        return self.fine_tune_questions(res)

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
        question = torch.stack(question).T # why am I doing stack question?
        print_sentence(question[0], self.word_index_to_word)
        f_q = self.forward_question(question) # should be 256, 256
        im_vector = self.forward_image(image) # should be 256, 256
        question_image_vector = torch.cat((f_q, im_vector), 1)
        logits = self(question_image_vector)
        # loss is just cross-entropy loss between answer and question_image vector   
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, answer)
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
        # answers = [1, 500, 5000, 15000, ...] glove[15000] = "red"
        # answers = [1,... , 3000]
        return Adam(self.parameters(), lr=self.op.learning_rate)