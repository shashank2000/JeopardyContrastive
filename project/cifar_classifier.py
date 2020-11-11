from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
import torch.nn as nn
from model_im_q_a import JeopardyModel2
from model import JeopardyModel
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import Adam
from pytorch_lightning.loggers import WandbLogger

image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CIFAR10Classifier(pl.LightningModule):
    
    def __init__(self, vocab_sz, jeop_model_path, model_type, parent_config, config):
        # seems like I need to remember the vocab size for some reason?
        super().__init__()
    
        if model_type == "inv":
            self.main_model == JeopardyModel2.load_from_checkpoint(jeop_model_path, vocab_sz=vocab_sz, config=parent_config)
        else:
            self.main_model = JeopardyModel.load_from_checkpoint(jeop_model_path, vocab_sz=vocab_sz, config=parent_config)
    
        self.main_model.freeze()
        self.picture_model = self.main_model.image_feature_extractor
        self.fine_tune = nn.Linear(128, 10) # logistic regression
        self.test_accuracy =  pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

        op = config.optim_params
        self.dataloaders = cifar10(op.batch_size)
        self.learning_rate = op.learning_rate

    def forward(self, x):
        x = self.picture_model(x)
        x = self.fine_tune(x)
        return nn.functional.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.nll_loss(logits, y)
        self.log('val_loss', loss)
        acc = self.val_accuracy(logits, y) 
        self.log('val_acc', acc)
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

    def train_dataloader(self):
        return self.dataloaders[0]
    
    def val_dataloader(self):
        return self.dataloaders[1]
    
    def test_dataloader(self):
        return self.dataloaders[2]

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    
def cifar10(bs):
    train_dataset = datasets.CIFAR10(
        root="/data5/wumike/cifar10",
        train=True,
        transform=image_transforms
    )
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(0.2*dataset_size)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=bs, num_workers=48)
    val_dataloader = DataLoader(train_dataset, sampler=val_sampler, batch_size=bs, num_workers=48)

    test_dataset = datasets.CIFAR10(
        root="/data5/wumike/cifar10",
        train=False,
        transform=image_transforms
    )
    test_dataloader = DataLoader(test_dataset, batch_size=bs, num_workers=48)

    return train_dataloader, val_dataloader, test_dataloader