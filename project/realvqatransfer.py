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
from loss_func.vqa_transfer_obj import VQATransferObjective
from utils.model_utils import pretrain_optimizer
import torch.nn.functional as F

def get_word(answer, pred, word_dict):
    '''
        Utility function as a sanity check. 
    '''
    for i, a in enumerate(answer):
        answer_word = word_dict[a.item()]
        pred_word = word_dict[torch.argmax(pred[i]).item()]
        print("answer is {}, pred is {}".format(answer_word, pred_word))

class NNJeopardyTest(pl.LightningModule):
    ''''
        We find the closest neighbor, and also return top-5 accuracy by using the topK function.
        answer_tokens: the list of tokens that are answers. If answer_tokens = [1,3,16] then tokens 1, 3 and 16 are all valid answers.
        This list is used in the construction of the answer embedding bank.
        word_index_to_word: for debugging purposes
    '''
    def __init__(self, main_model_path, parent_config, config, vocab_sz=None, answer_tokens=None, word_index_to_word=None):
        super().__init__()
        self.save_hyperparameters('main_model_path', 'parent_config', 'config', 'vocab_sz')
        if parent_config.system == "inverse-jeopardy":
            self.main_model = JeopardyModel2.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
        else:
            self.main_model = JeopardyModel.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
        
        self.op = config.optim_params
        self.mp = config.model_params
        self.word_index_to_word = word_index_to_word
        self.num_possible_tokens = config.num_classes
        
        # all pretrained weights are frozen
        self.main_model.freeze()

        self.q_rnn = self.main_model.rnn
        self.q_embed = self.main_model.i_h # learned embedding layer
        
        self.h = self.main_model.h 
        self.ans_final = self.main_model.ans_final # 128d

        self.resnet = self.main_model.image_feature_extractor # outputs in the right dimension because we modified the fc in the pretrain task
        # we want post-pool layer, and want to get rid of the projection head, which is currently image_feature_extractor.fc
        
        self.test_accuracy =  pl.metrics.Accuracy()
        self.test_accuracy_top_k =  pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy() # to check for overfitting!

        self.h_o = self.main_model.h_o # outputs a 256 dim vector for question embedding
        
        # finetuning the concatenated answer-image vector
        self.fine_tune_answer_image = nn.Linear(self.mp.q_dim, self.mp.q_dim)
        self.fine_tune_question = nn.Linear(self.mp.q_dim, self.mp.q_dim)

        self.answer_tokens = answer_tokens
        self.answer_embed_bank = None
    
    def _build_embed_bank(self):
        print("building answer embed bank")
        with torch.no_grad():
            self.answer_tokens = torch.tensor(self.answer_tokens, device=self.device)
            maxIndex = max(self.answer_tokens)
            self.answer_embed_bank = torch.zeros(maxIndex + 1, self.mp.ans_dim, device=self.device)   
            for i in self.answer_tokens:
                # answer_embed_bank[i] is the embedding for token i in the vocabulary
                self.answer_embed_bank[i] = self.forward_answer(i)

            self.answer_embed_bank.requires_grad = False

    def forward_image(self, x):
        x = self.resnet(x)
        return x
        
    def forward_question(self, x):
        # we have a question as input
        res, h = self.q_rnn(self.q_embed(x), self.h)
        self.h = h[0].detach(), h[1].detach()
        # res is currently of the form (10, 256, 50); we only want the last word predictions
        res = self.h_o(res)[-1]
        return res
    
    def forward_answer(self, x):
      # just a linear layer over the embeddings to begin with
      x = self.q_embed(x)
      return self.ans_final(x)
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx=batch_idx)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, testing=True)
        self.log("test_loss", loss)
        return loss

    def shared_step(self, batch, testing=False, batch_idx=0):
        # self.answer_embed_bank contains the pretrained answer embeddings
        if self.answer_embed_bank is None:
            self._build_embed_bank()

        question, image, answer = batch
        
        question = torch.stack(question)
        f_q = self.forward_question(question) # 10, 256, 256
        f_q = self.fine_tune_question(f_q)
        im_vector = self.forward_image(image)

        answer_emb = self.forward_answer(answer)
        answer_image_vector = torch.cat((answer_emb, im_vector), 1)
        answer_image_vector = self.fine_tune_answer_image(answer_image_vector)
        
        loss = VQATransferObjective(f_q, answer_image_vector, self.answer_embed_bank, im_vector, k=self.mp.loss_k, t=self.mp.loss_t).get_loss(self.answer_tokens)
        
        batch_size = im_vector.shape[0]
        pred = torch.zeros(batch_size, self.answer_embed_bank.shape[0], device=self.device)
        
        with torch.no_grad(): 
            '''
                We are building the answer_image_embed bank here; the weird construction is mostly due to memory constraints. 
                Once we build the answer_image_embed_bank, we make the prediction vector for question i; the best answer_image/question
                match is taken to be the answer_image corresponding to the question.
            '''
            image_bank = torch.cat([im_vector.unsqueeze(1)] * self.answer_embed_bank.shape[0], dim=1)
            answer = F.one_hot(answer, num_classes=len(self.answer_embed_bank))
        
            for i in range(len(pred)):
                repeated_indiv_im_vec = image_bank[i]
                answer_image_embed_bank = torch.cat((self.answer_embed_bank, repeated_indiv_im_vec), dim=1) # (16000, 256)
                answer_image_embed_bank = self.fine_tune_answer_image(answer_image_embed_bank)
                answer_image_embed_bank = F.normalize(answer_image_embed_bank, dim=1)
                # prediction for the ith image
                pred[i] = f_q[i] @ answer_image_embed_bank.T # (256) * (256, 16000) = (16000)

        if not testing:
            if batch_idx % 10 == 0:
                breakpoint()
                get_word(answer, pred, self.word_index_to_word)
            acc = self.train_accuracy(pred, answer) # if pred is [0.1, 0.7, 10] and answer is 2, it should fire
            self.log('train_acc', acc)
        else:
            acc = self.test_accuracy(pred, answer) # pred is shape (batch_size, 3000) while answer is just one answer per question_image, but should be fine
            top_k_acc = None
            vec = torch.topk(pred, self.mp.top_k).indices
            for i, a in enumerate(answer):
                found = False
                for v in vec[i]:
                    if v == a:
                        top_k_acc = self.test_accuracy_top_k(a, a)
                        found = True
                        break
                if not found:
                    # see https://github.com/PyTorchLightning/pytorch-lightning/blob/bf7c28cd54b5df05bf6c97614ad9cea3d001c105/pytorch_lightning/metrics/classification/accuracy.py
                    top_k_acc = self.test_accuracy_top_k(a+1,a)
            self.log('test_acc', acc)
            self.log('top_k_acc', top_k_acc)
        return loss

    def configure_optimizers(self):
        return pretrain_optimizer(self.parameters(), self.op.momentum, self.op.weight_decay, self.op.learning_rate, lars=False)