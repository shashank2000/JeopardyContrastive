import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiamSystem(nn.Module):
    def __init__(self, config, encoder):
        # config is for dimension stuff
        self.model = self.get_model()
        self.config = config
        self.encoder = encoder

    def get_model(self):
        out_dim = self.config.model_params.out_dim
        hid_dim = self.config.model_params.hidden_dim
        model = SiameseArm(self.encoder, hid_dim, hid_dim, out_dim,
                           posterior_head=self.config.model.posterior_head)
        return model

    def forward(self, questions=None, images=None, answers=None):
        _, z, h = self.model(
            questions=questions, 
            images=images, 
            answers=answers
            )
        return z, h

    def get_similarity(self, a, b):
        b = b.detach()  # stop gradient of backbone + projection mlp
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = -1 * (a * b).sum(-1).mean()
        return sim

    def get_loss(self, batch, train=True, **kwargs):
        question, image, answer = batch
        z1, h1 = self.forward(images=image, answers=answer)
        z2, h2 = self.forward(questions=question)
        loss = self.get_similarity(h1, z2) / 2 + self.get_similarity(h2, z1) / 2
        return loss


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):

    def __init__(self, encoder, input_dim, hidden_size, output_dim):
        super().__init__()
        # if it sees image input, puts it through ResNet, else puts it through linear layer
        self.encoder = encoder  
        self.projector = MLP(input_dim, hidden_size, output_dim)
        self.predictor = MLP(output_dim, hidden_size, output_dim)

    def forward(self, images=None, answers=None, questions=None):
        y = self.encoder(images=images, answers=answers, questions=questions)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h