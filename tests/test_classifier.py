from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from project.lit_classifier_main import LitClassifier

class CIFAR10Classifier(pl.LightningModule):
"""
Idea: we start with pretrained model that is the argumet, we are tetsting how good the representation is. Run a couple of epochs. Freeze the rep.	
"""

    def __init__(self, pretrained):
        self.main_model = # load from checkpoint the trained version of it
        self.fine_tune = 10

    def forward(self, x):
        x = self.main_model.image_feature_extractor(x)
    

        # fine tune to make it so that it outputs only 10 classes
# TODO: for downstream task testing
def test_lit_classifier():
    seed_everything(1234)

    model = LitClassifier()
    train, val, test = mnist()
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, train, val)

    results = trainer.test(test_dataloaders=test)
    assert results[0]['test_acc'] > 0.7
