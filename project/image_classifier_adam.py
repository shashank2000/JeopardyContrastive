from pytorch_lightning import Trainer, seed_everything
from dataset.mscoco import BaseMSCOCO
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from pytorch_lightning.utilities import AMPType
from model_im_q_a import JeopardyModel2
from jointobjective import JeopardyModelv2Joint
from model import JeopardyModel
from additionmodel import JeopardyAddModel
from torchvision import transforms, datasets
from torch.optim import SGD, Adam
from pytorch_lightning.loggers import WandbLogger
from baseline_simclr import UpperBoundModel
from model_utils import pretrain_optimizer, pretrain_scheduler
from test_utils import get_main_model
from v2model import JeopardyModelv2

class SimpleClassifierAdam(pl.LightningModule):
    
    def __init__(self, main_model_path, parent_config, config, vocab_sz=None, num_samples=1000):
        '''
        vocab_sz: only relevant if pretrained on VQA, size of the vocabulary
        main_model_path: path to checkpoint that we want to test
        model_type: type of the pretrained model - 'inv', 'regular' or 'coco'
        parent_config: path to the parent's config file so we won't have hiccups while loading from checkpoint
        config: config file for this run

        No finetuning, only linear evaluation. "Using pretraining hyperparams with LARS yields similar results" (B.6)

        TODO: Add a flag to train with B5 finetuning hyperparams instead

        From the paper, if improvements needed for CIFAR-10:
            As our goal is not to optimize CIFAR-10 performance, but rather to provide further confirmation of our observations
            on ImageNet, we use the same architecture (ResNet-50) for CIFAR-10 experiments. Because CIFAR-10 images are much
            smaller than ImageNet images, we replace the first 7x7 Conv of stride 2 with 3x3 Conv of stride 1, and also remove the first
            max pooling operation. For data augmentation, we use the same Inception crop (flip and resize to 32x32) as ImageNet, and
            color distortion (strength=0.5), leaving out Gaussian blur. We pretrain with learning rate in {0.5, 1.0, 1.5}, temperature in
            {0.1, 0.5, 1.0}, and batch size in {256, 512, 1024, 2048, 4096}. The rest of the settings (including optimizer, weight decay,
            etc.) are the same as our ImageNet training.
        '''
        
        super().__init__()

        # or load weights mapping all weights from GPU 1 to GPU 0 ...
        self.main_model = get_main_model()
        self.main_model.freeze()
        self.resnet = self.main_model.image_feature_extractor
        # confirm it is indeed frozen here
        
        # post pool features taken; getting rid of projection head
        # self.block_forward = nn.Sequential(
        #     nn.Dropout(p=config.dropout_p),
        # )
        self.resnet.fc = nn.Linear(512, config.num_classes)
 # logistic regression, and that's it!

        self.test_accuracy =  pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy() # to check for overfitting!
        self.op = config.optim_params

    def forward(self, x):
        x = self.resnet(x)
        return nn.functional.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.nll_loss(logits, y)
        self.log('train_loss', loss)
        acc = self.train_accuracy(logits, y) 
        self.log('train_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        # also calculate accuracy
        x, y = batch
        logits = self(x)
        loss = nn.functional.nll_loss(logits, y)
        acc = self.test_accuracy(logits, y) 
        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.op.learning_rate)