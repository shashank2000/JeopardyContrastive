from image_classifier import SimpleClassifier
from run_script import seed_everything
import pytorch_lightning as pl 
from utils.setup import process_config
from baseline_data_module import BaselineDataModule
from data_module import VQADataModule
from vqatransfer import DumbJeopardyTest
from realvqatransfer import NNJeopardyTest
from vqabaseline import BaselineVQA

def run(vocab_sz, checkpoint, config_path, hundred_epochs=False, parent_config=None, gpu_device=None):
    '''
    TODO: WE ONLY USE RANDOM CROPS AND FLIPS AS AUGMENTATIONS IN TRANSFER! THIS ISN'T HAPPENING YET
    '''
    config = process_config(config_path)
    if not gpu_device:
        gpu_device = config.gpu_device
    seed_everything(config.seed, use_cuda=config.cuda)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=-1, # could be 5, and that would work fine
        period=1,
    )
    
    num_epochs = config.num_epochs
    if hundred_epochs:
        num_epochs = 100
    
    run_name = config.run_name or "testing " + checkpoint
    wandb_logger = pl.loggers.WandbLogger(name=run_name, config=config, project=config.exp_name)
    # if we are at 100 epochs pretraining, run for 100 epochs 
    # we are using the same transformations as we did in the pretraining task, but this time for regularization etc   
    model, dm = None, None
    if config.mtype == "vqa":
        # TODO: num_samples stuff here
        dumb_transf = ("fancier_jeop_test" not in config.exp_name) # everything else is dumb
        answer_classes = config.answer_classes
        dm = VQADataModule(num_answers=answer_classes, batch_size=config.optim_params.batch_size, num_workers=config.num_workers, dumb_transfer=dumb_transf, transfer=True)
        # the models get the number of answer classes from the config file
        if config.exp_name == "vqa_baseline":
            model = BaselineVQA(checkpoint, process_config(parent_config), config, word_index_to_word=dm.idx_to_word)
        elif config.exp_name == "dumb_jeop_test":
            model = DumbJeopardyTest(checkpoint, process_config(parent_config), config, vocab_sz=vocab_sz)
        else:
            model = NNJeopardyTest(checkpoint, process_config(parent_config), config, vocab_sz=vocab_sz, answer_tokens=dm.train_dataset.answer_tokens, word_index_to_word=dm.idx_to_word)
    else:
        dm = BaselineDataModule(batch_size=config.optim_params.batch_size, num_workers=config.num_workers, dataset_type=config.mtype)
        # add len(dm) in here to find num_samples and pass it into the model
        model = SimpleClassifier(checkpoint, process_config(parent_config), config, vocab_sz=vocab_sz, num_samples=len(dm.train_dataset))
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=[4,5,6,7],
        distributed_backend='ddp',
        max_epochs=num_epochs,
        checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.continue_from_checkpoint,
        logger=wandb_logger
    )

    trainer.fit(model, dm)
    results = trainer.test()
    print(results[0]['test_acc'])
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='config/cifar10transfer.json')
    parser.add_argument('checkpoint', type=str, default=None)
    parser.add_argument('vocab_size', type=int, default=20541)
    parser.add_argument('parent_config', type=str, default="config/jeopardy_model.json") # to pass in while loading model
    parser.add_argument('--gpu-device', type=int, default=8)
    parser.add_argument('-l', action='store_true', 
    help="run a larger number of epochs for this run")
    args = parser.parse_args()
    run(
        vocab_sz=args.vocab_size, 
        checkpoint=args.checkpoint, 
        config_path=args.config, 
        gpu_device=args.gpu_device, 
        parent_config=args.parent_config,
        hundred_epochs=args.l
    )