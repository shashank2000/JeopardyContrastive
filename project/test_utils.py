from model_im_q_a import JeopardyModel2
from v2model import JeopardyModelv2
from v3model import v3Model
from v3modelcross import v3ModelCross
from model import JeopardyModel
from simsiam import SimSiamJeopardy

def get_main_model(parent_config, main_model_path, vocab_sz):
    if parent_config.system == "inverse-jeopardy":
        main_model = JeopardyModel2.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
    elif parent_config.system == "v2-jeopardy":
        main_model = JeopardyModelv2.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
    elif parent_config.system ==  "symmetric-jeopardy":
        main_model = v3Model.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
    elif parent_config.system == "symmetric-jeopardy-cross":
        main_model = v3ModelCross.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
    elif parent_config.system == "simsiam":
        main_model = SimSiamJeopardy.load_from_checkpoint(main_model_path, config=parent_config)    
    else:
        main_model = JeopardyModel.load_from_checkpoint(main_model_path, vocab_sz=vocab_sz, config=parent_config)
    return main_model

from utils.setup import process_config


def get_downstream_model_and_dm(config, parent_config, checkpoint, vocab_sz):
    model, dm = None, None
    
    if config.mtype == "vqa":
        # TODO: num_samples stuff here
        dumb_transf = ("fancier_jeop_test" not in config.exp_name) # everything else is dumb
        answer_classes = config.answer_classes
        dm = VQADataModule(num_answers=answer_classes, batch_size=config.optim_params.batch_size, num_workers=config.num_workers, dumb_transfer=dumb_transf, transfer=True)
        # the models get the number of answer classes from the config file
        if config.system == "vqa_baseline":
            # TODO: fix vqa baseline json to reflect this
            model = BaselineVQA(checkpoint, process_config(parent_config), config, word_index_to_word=dm.idx_to_word)
        elif config.system == "djeopardy":
            model = DumbJeopardyTest(checkpoint, process_config(parent_config), config, vocab_sz=vocab_sz)
        else:
            model = NNJeopardyTest(checkpoint, process_config(parent_config), config, vocab_sz=vocab_sz, answer_tokens=dm.train_dataset.answer_tokens, word_index_to_word=dm.idx_to_word)
    else:
        dm = BaselineDataModule(batch_size=config.optim_params.batch_size, num_workers=config.num_workers, dataset_type=config.mtype)
        # add len(dm) in here to find num_samples and pass it into the model
        if config.adam:
            model = SimpleClassifierAdam(checkpoint, process_config(parent_config), config,  vocab_sz=vocab_sz, num_samples=len(dm.train_dataset))
        else:
            model = SimpleClassifier(checkpoint, process_config(parent_config), config, vocab_sz=vocab_sz, num_samples=len(dm.train_dataset))
    
    return model, dm

from data_module import VQADataModule
from vqabaseline import BaselineVQA
from vqatransfer import DumbJeopardyTest
from realvqatransfer import NNJeopardyTest
from baseline_data_module import BaselineDataModule
from image_classifier_adam import SimpleClassifierAdam
from image_classifier import SimpleClassifier
