from image_classifier import SimpleClassifier
from run_script import seed_everything
import pytorch_lightning as pl 
from utils.setup import process_config

def run(vocab_sz, checkpoint, config_path, parent_config=None, gpu_device=None, model_type="regular"):
    # TODO: VQA transfer task
    config = process_config(config_path)
    if not gpu_device:
        gpu_device = config.gpu_device
    seed_everything(config.seed, use_cuda=config.cuda)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=-1, # could be 5, and that would work fine
        period=1,
    )
    
    wandb_logger = pl.loggers.WandbLogger(name="testing " + checkpoint, config=config, project=config.exp_name)    
    model = SimpleClassifier(checkpoint, model_type, process_config(parent_config), config, vocab_sz=vocab_sz)
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=[gpu_device],
        max_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.continue_from_checkpoint,
        logger=wandb_logger
    )

    trainer.fit(model)
    results = trainer.test()
    print(results[0]['test_acc'])
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='config/cifar10transfer.json')
    parser.add_argument('checkpoint', type=str, default=None)
    parser.add_argument('vocab_size', type=int, default=20541)
    parser.add_argument('parent_config', type=str, default="config/jeopardy_model.json") # to pass in while loading model
    parser.add_argument('model_type', type=str, default="regular")
    parser.add_argument('--gpu-device', type=int, default=0)
    args = parser.parse_args()
    run(
        vocab_sz=args.vocab_size, 
        checkpoint=args.checkpoint, 
        model_type=args.model_type, 
        config_path=args.config, 
        gpu_device=args.gpu_device, 
        parent_config=args.parent_config
    )