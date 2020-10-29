from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from model import JeopardyModel
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from pytorch_lightning.loggers import WandbLogger

# Steps:
# 1. test with existing checkpoint, and we can see what kind of accuracy we are getting
# checkpoint path is an argument

wandb_logger = WandbLogger(name='testing_the_representation_with_SGD_50_epochs_starting_from_SGD_3q915wgy',project='contra-pytorchlightning')

PATH = "/mnt/fs5/shashank2000/contra-pytorchlightning/kosxmjr3/checkpoints/epoch=199.ckpt"
image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class CIFAR10Classifier(pl.LightningModule):
    
    def __init__(self, jeop_model_path, learning_rate=0.03):
        # seems like I need to remember the vocab size for some reason?
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.main_model = JeopardyModel.load_from_checkpoint(jeop_model_path, vocab_sz=20541)
        self.main_model.freeze() # make sure this works..
        self.picture_model = self.main_model.image_feature_extractor
        self.fine_tune = nn.Linear(128, 10) # logistic regression
        self.train_accuracy = pl.metrics.Accuracy()
        self.test_accuracy =  pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.dataloaders = cifar10()

    def forward(self, x):
        x = self.picture_model(x)
        return self.fine_tune(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss)
        acc = self.train_accuracy(logits, y) 
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss)
        acc = self.val_accuracy(logits, y) 
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        # also calculate accuracy
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
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
    
    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)

    
def cifar10():
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

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16, num_workers=48)
    val_dataloader = DataLoader(train_dataset, sampler=val_sampler, batch_size=16, num_workers=48)

    test_dataset = datasets.CIFAR10(
        root="/data5/wumike/cifar10",
        train=False,
        transform=image_transforms
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=48)

    return train_dataloader, val_dataloader, test_dataloader


def test_lit_classifier(checkpoint=PATH):
    # collect args here and pass it into the model
    model = CIFAR10Classifier(checkpoint)
    
    trainer = Trainer(resume_from_checkpoint="/mnt/fs5/shashank2000/contra-pytorchlightning/3q915wgy/checkpoints/epoch=20.ckpt", profiler="simple", logger=wandb_logger, max_epochs=50, default_root_dir="/mnt/fs5/shashank2000/", gpus=[9])
    trainer.fit(model)
    results = trainer.test()
    print(results[0]['test_acc'])
    assert results[0]['test_acc'] > 0.7

# find args, pass it in here
# find args, pass it in here
import sys
# Pass in the filepath of the checkpoint of the trained VQA/Jeopardy model that you'd like to test.
if len(sys.argv) > 2:
    # no exception handling with invalid path, assuming that's okay
    test_lit_classifier(sys.argv[2])
else:
    test_lit_classifier()
