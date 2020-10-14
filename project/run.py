from data_module import VQADataModule
from PIL import ImageFile
import pytorch_lightning as pl
from model import JeopardyModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# add gpus=1 if training on a GPU; progress_bar_refresh_rate=50 if on Jupyter notebook
trainer = pl.Trainer(max_epochs=200, progress_bar_refresh_rate=1) 
dm = VQADataModule(batch_size=16)

model = JeopardyModel(dm.vl)

trainer.fit(model, dm)
# trainer.test(data_module=dm)