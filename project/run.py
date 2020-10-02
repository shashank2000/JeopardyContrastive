from data_preprocess import train_loader, val_loader, batch_size
from PIL import ImageFile
import pytorch_lightning as pl
from model import JeopardyModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=50) # add gpus=1 if training on a GPU

model = JeopardyModel(22805, batch_size)
trainer.fit(model, train_loader, val_loader)
