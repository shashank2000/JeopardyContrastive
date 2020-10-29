from data_module import VQADataModule
from PIL import ImageFile
import pytorch_lightning as pl
from model import JeopardyModel
from pytorch_lightning.loggers import WandbLogger
import subprocess

PROJECT_NAME = "contra_pytorchlightning"
wandb_logger = WandbLogger(name='from checkpoint with different hparams',project=PROJECT_NAME)

CHECKPOINT_BASE_PATH = "/mnt/fs5/shashank2000/" + PROJECT_NAME + "/{}/checkpoints/epoch={}.ckpt" 
class RealTimeEvalCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        print(trainer.current_epoch)
        if trainer.current_epoch % 2 == 0:
            version = pl_module.logger.version
            breakpoint()
            subprocess.Popen(["python", "new_test_class.py", CHECKPOINT_BASE_PATH.format(version, trainer.current_epoch)])

ImageFile.LOAD_TRUNCATED_IMAGES = True

# add gpus=1 if training on a GPU; progress_bar_refresh_rate=50 if on Jupyter notebook, add logger
trainer = pl.Trainer(callbacks=[RealTimeEvalCallback()], resume_from_checkpoint="/mnt/fs5/shashank2000/contra-pytorchlightning/tdcjm571/checkpoints/epoch=3.ckpt", gpus=[0], default_root_dir='/mnt/fs5/shashank2000/', max_epochs=200, progress_bar_refresh_rate=1, logger=wandb_logger) 
dm = VQADataModule(batch_size=256)
model = JeopardyModel(vocab_sz=dm.vl)
trainer.fit(model, dm)
# trainer.test(data_module=dm)
