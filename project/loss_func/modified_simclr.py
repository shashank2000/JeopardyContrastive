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
        witness_pos = torch.sum(self.outputs1 * self.outputs2, dim=1)
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        witness_partition = self.outputs1 @ outputs12.T
        witness_partition = torch.logsumexp(witness_partition / self.t, dim=1)
        loss = -torch.mean(witness_pos / self.t - witness_partition)
        similar_but_wrong_questions = get_from_cluster(self.outputs1) # or maybe it makes more sense to pass in to the loss function? 
        # similar_but_wrong_questions is a k x m matrix, self.outputs2 is the image-answers in the batch
        similar_but_wrong_questions = l2_normalize(similar_but_wrong_questions, dim=1)
        loss -= torch.mean(similar_but_wrong_questions @ self.outputs2.T) # the higher this value, the more dissimilar they are, which is what we want
        return loss