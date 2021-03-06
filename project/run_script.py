import os
import random, torch, numpy
from utils.setup import process_config
import pytorch_lightning as pl
from model import JeopardyModel
from model_im_q_a import JeopardyModel2
from data_module import VQADataModule
from baseline_data_module import BaselineDataModule
from baseline_simclr import UpperBoundModel
import subprocess
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RealTimeEvalCallback(pl.Callback):
    def __init__(self, checkpoint_dir, downstream_task_config, vocab_sz=100, parent_config=None, d2=None, mtype="regular"):
        self.checkpoint_dir = checkpoint_dir
        self.downstream_task_config = downstream_task_config
        self.vocab_sz = str(vocab_sz)
        self.parent_config = parent_config
        self.downstream2=d2
        self.mtype = mtype

    def on_validation_epoch_end(self, trainer, pl_module):
        cur = trainer.current_epoch
        if cur % 10 == 0 or cur == 1:
            cur_checkpoint = os.path.join(self.checkpoint_dir, "epoch="+str(cur)+".ckpt")
            commands = ["python", "test_representation.py", self.downstream_task_config, cur_checkpoint, self.vocab_sz, self.parent_config]
            commands.extend([self.mtype, "--gpu-device", "0"])
            subprocess.Popen(commands)
            if self.downstream2:
                # this only on COCO with Jeopardy Model, not CIFAR10 on baseline
                commands = ["python", "test_representation.py", self.downstream_task_config, cur_checkpoint, self.vocab_sz, self.parent_config, self.mtype, "--gpu-device", "5"]
                subprocess.Popen(commands)

def run(config_path, gpu_device=None):
    config = process_config(config_path)
    seed_everything(config.seed, use_cuda=config.cuda)

    # Perhaps I can change this so that it doesn't save each epoch's results?
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=-1, # could be 5, and that would work fine
        period=1,
    )
    
    wandb_logger = pl.loggers.WandbLogger(name=config.run_name, project=config.exp_name)
    dm = BaselineDataModule(batch_size=config.optim_params.batch_size) if config.system == "coco-pretraining" else VQADataModule(batch_size=config.optim_params.batch_size)
    mp = config.model_params
    
    model = None
    eval_realtime_callback = None
    if config.system == "coco-pretraining":
        model = UpperBoundModel(config)
        eval_realtime_callback = RealTimeEvalCallback(checkpoint_dir=config.checkpoint_dir, downstream_task_config=config.downstream_task_config, parent_config=config_path, mtype="coco")
    else:
        if config.system == "InverseJeopardy":
            model = JeopardyModel2(dm.vl, mp.im_vec_dim, mp.ans_dim, mp.question_dim, mp.n_hidden, mp.n_layers, config.optim_params, mtype="inv")
        else:  
            model = JeopardyModel(dm.vl, config)
        eval_realtime_callback = RealTimeEvalCallback(config.checkpoint_dir, config.downstream_task_config, dm.vl, config_path, d2=config.downstream_task2_config)

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=[gpu_device],
        max_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        callbacks=[eval_realtime_callback],
        resume_from_checkpoint=config.continue_from_checkpoint,
        logger=wandb_logger
    )

    trainer.fit(model, dm)


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='config/jeopardy_model.json')
    parser.add_argument('--gpu-device', type=int, default=None)
    args = parser.parse_args()
    run(args.config, gpu_device=args.gpu_device)