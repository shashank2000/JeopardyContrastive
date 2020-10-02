from data_preprocess import train_loader, val_loader, batch_size
from PIL import ImageFile
import pytorch_lightning as pl
from model import JeopardyModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# add gpus=1 if training on a GPU; progress_bar_refresh_rate=50 if on Jupyter notebook
trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=1) 

model = JeopardyModel(22805, batch_size)
trainer.fit(model, train_loader, val_loader)
