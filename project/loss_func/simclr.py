import torch

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

class SimCLR(object):

    def __init__(self, outputs1, outputs2, t=0.07, **kwargs):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t

    def get_loss(self):
        # sim(u,v) denotes the cosine similarity between the two vectors
        # dot product of the two vectors
        witness_pos = torch.sum(self.outputs1 * self.outputs2, dim=1)
        # concatenating the two
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        # answer_image vector multiplied by the transpose of the concatenated vector
        witness_partition = self.outputs1 @ outputs12.T 
        witness_partition = torch.logsumexp(witness_partition / self.t, dim=1)
        # final loss is computed across all positive pairs in a minibatch, and we take average
        loss = -torch.mean(witness_pos / self.t - witness_partition) # sum of everything minus (i, i)
        # returns mean value of each row of the input tensor
        return loss