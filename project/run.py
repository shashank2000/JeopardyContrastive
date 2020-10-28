from data_module import VQADataModule
from PIL import ImageFile
import pytorch_lightning as pl
from model import JeopardyModel
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(name='Adam-32-0.001',project='contra-pytorchlightning')

ImageFile.LOAD_TRUNCATED_IMAGES = True

# add gpus=1 if training on a GPU; progress_bar_refresh_rate=50 if on Jupyter notebook, add logger
trainer = pl.Trainer(max_epochs=200, progress_bar_refresh_rate=1, logger=wandb_logger) 
dm = VQADataModule(batch_size=16)

model = JeopardyModel(dm.vl)

trainer.fit(model, dm)
# trainer.test(data_module=dm)