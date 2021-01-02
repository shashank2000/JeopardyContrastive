import torch 
import torch.nn.functional as F

class VQATransferObjective(object):
  '''
    We are using this loss to come up with the best possible answer embedding, given a question
    and image pair.
    k: number of images we want to use while sampling
    t: temperature 
    question_emb: the question vector
    answer_emb: the answer vector in question (this is training)
    answer_emb_bank: all the answer vectors, so of size x * emb_dim where x is the total number of answers 
  '''
  def __init__(self, question_emb, answer_image_emb, answer_emb_bank, image_emb, k=4096, t=0.07, **kwargs):
    self.k, self.t = k, t
    self.question_emb = F.normalize(question_emb, dim=1)  # batch_size x emb_dim
    self.answer_image_emb = F.normalize(answer_image_emb, dim=1) # batch_size x emb_dim
    self.answer_emb_bank = answer_emb_bank # num_answer_classes x emb_dim; we normalize later
    self.image_emb_bank =  torch.cat([image_emb.unsqueeze(1)] * self.k, dim=1)# we normalize these later once we've concatenated with answer_emb_bank selectively 
    self.device = self.question_emb.device

  def get_loss(self, non_zero_indices, *args, **kwargs):
    # we start with just using all 30k negatives, no sampling complications
    batch_size = self.question_emb.size(0) # or .shape[0]

    score = torch.sum(self.question_emb * self.answer_image_emb, dim=1)
    
    if self.k < self.answer_emb_bank.size(0):
      # hacky way to get out of actually changing numericalization logic in dataset
      # retraining the pretrained embeddings would take too long :(
      noise_idxs = torch.randint(0, non_zero_indices.size(0), 
                                  (batch_size, self.k))
      noise_idxs = noise_idxs.long().to(self.device).view(-1)
      noise_idxs = torch.index_select(non_zero_indices, 0, noise_idxs)
      with torch.no_grad():
          flat_idxs = noise_idxs.view(-1)
          answer_emb_bank = torch.index_select(self.answer_emb_bank, 0, flat_idxs)
          answer_emb_bank = answer_emb_bank.view(batch_size, self.k, -1) # batch_size, self.k, answer_emb_dim
          # we concatenate images corresponding to the batch
          answer_image_emb_bank = torch.cat((answer_emb_bank, self.image_emb_bank), dim=2) # batch_size, self.k, answer_emb_dim + img_emb_dim
          answer_image_emb_bank = F.normalize(answer_image_emb_bank, dim=2) # normalize each answer-image embedding
          self.answer_image_emb_bank = answer_image_emb_bank
    
    norm = torch.einsum('ij,ikj->ik', [self.question_emb, self.answer_image_emb_bank]).to(self.device)
    # we need answer 1's embedding to be at answer_emb_bank[1]
    norm = torch.logsumexp(norm / self.t, dim=1)
    loss = -torch.mean(score / self.t - norm) 
    return loss