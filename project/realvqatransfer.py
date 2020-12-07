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
    # word_dict is passed in by the dataloader
    for i, a in enumerate(answer):
        answer_word = word_dict[a.item()]
        pred_word = word_dict[torch.argmax(pred[i]).item()]
        print("answer is {}, pred is {}".format(answer_word, pred_word))

class NNJeopardyTest(pl.LightningModule):
    ''''
        We find the closest neighbor, and also return top-5 accuracy by using the topK function.
    '''

    def __init__(self, main_model_path, parent_config, config, vocab_sz=None, answer_tokens=None, word_index_to_word=None):
        # we do everything exactly the same way as the main task, except there's no projection head
        # do weighted voting instead?
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
        # hidden state for main model would be whatever the end hidden state of the last batch was, pretty sure this can be made 0 as well
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
            # 16000 possible values for i where answer = [i] 
            # 
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
        # [19]
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
        # build this once
        if self.answer_embed_bank is None:
            self._build_embed_bank()

        # we've got to update the answer embed bank once per batch to reflect the new answer embeddings because 
        # of changed weights on the answers
        question, image, answer = batch
        question = torch.stack(question)
        f_q = self.forward_question(question) # its 10, 256, 256
        f_q = self.fine_tune_question(f_q)
        im_vector = self.forward_image(image)

        answer_emb = self.forward_answer(answer)
        answer_image_vector = torch.cat((answer_emb, im_vector), 1)
        answer_image_vector = self.fine_tune_answer_image(answer_image_vector)
        # we have an answer embed bank, now we concatate the image to each answer
        # loss is just cross-entropy loss between answer and question_image vector
        loss = VQATransferObjective(f_q, answer_image_vector, self.answer_embed_bank, im_vector, k=self.mp.loss_k, t=self.mp.loss_t).get_loss(self.answer_tokens)
        # dot product is linear, so it doesn't matter how we actually do this
        # let's make pred the dot product between question embedding and the image_answer bank (embed the same image in front of each answer)
        # we finally pick the best answer
        # could run profiler to see if moving to CPU is faster than this looping business
        batch_size = im_vector.shape[0]
        pred = torch.zeros(batch_size, self.answer_embed_bank.shape[0], device=self.device)
        
        # blue - 11, pink - 16000
        # 9500 - get rid of zeros
        # add torch nograd here
        with torch.no_grad(): 
            # try einstein summation   
            # build the answer embed bank here
            # f_q (256, 256); answer_image_embed_bank (16000, 256) -> final is just 16000

            # first we build the answer_image_embed_bank, which would be given by "merging" the answer_embed_bank with the image for each triplet
            # im_vector is (256, 128); answer_embed_bank is 16000, 128 => answer_image_embed_bank would be 256, 16000, 256 # on answer-image embedding for each answer
            # make an answer_image_embed_bank; 256, 16000, 256

            image_bank = torch.cat([im_vector.unsqueeze(1)] * self.answer_embed_bank.shape[0], dim=1)
            # if there were no memory constraints, we'd concatenate the image_bank with the answer_bank here; before this the answer bank needs to be made 256x in size
            # but this is too large, and so we use a for loop
            # answer_image_embed_bank = torch.cat([answer_embed_bank, image_bank], dim=2)
            # pred = torch.einsum('ab,akb->ak', [f_q, answer_image_embed_bank])
            
            # answer_embed_bank: (16000, 128) -> (256, 16000, 128)
            # (256, 128) - im_vector -> 256, 16000, 128
            # image_answer_embed_bank
            # answer_image_embed_bank -  (256, 16000, 128)
            
            # NCE without normalize
            # NCE, k = 16000
            # normalize on line 162

            for i in range(len(pred)):
                repeated_indiv_im_vec = image_bank[i]
                answer_image_embed_bank = torch.cat((self.answer_embed_bank, repeated_indiv_im_vec), dim=1) # (16000, 256)
                answer_image_embed_bank = self.fine_tune_answer_image(answer_image_embed_bank)
                # prediction for the ith image
                answer_image_embed_bank = F.normalize(answer_image_embed_bank, dim=1)
                pred[i] = f_q[i] @ answer_image_embed_bank.T # (256) * (256, 16000) = (16000)

        if not testing:
            if batch_idx % 10 == 0:
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
                    top_k_acc = self.test_accuracy_top_k(a+1,a)
            self.log('test_acc', acc)
            self.log('top_k_acc', top_k_acc)
        
        return loss

    def configure_optimizers(self):
        # DOES NOT USE LARS, USES THE 0.1x 120 160 STUFF - see Instance Discrimination paper
        return pretrain_optimizer(self.parameters(), self.op.momentum, self.op.weight_decay, self.op.learning_rate, lars=False)