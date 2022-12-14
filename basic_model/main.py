import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from data import DataPlatform
from model_selection import Selection
from trainer import Trainer
from test import Test
from kfold import KFold

def set_seeds(seed=random.randrange(1, 10000)):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # for faster training, but not deterministic

if __name__ == "__main__":
    ## Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--num_hidden_layer", type=int)
    parser.add_argument("--mx_token_size", type=int)
    parser.add_argument("--kfold_flag", type=bool)
    parser.add_argument("--kfold_k", type=int)
    parser.add_argument("--early_stopping_flag", type=bool)
    parser.add_argument("--only_clasifi_flag", type=bool)
    parser.add_argument("--only_reg_flag", type=bool)
    parser.add_argument("--reg_plus_clasifi_flag", type=bool)
    parser.add_argument("--reg_plus_multi_clasifi_flag", type=bool)
    parser.add_argument("--clasifi_2_clasifi_flag", type=bool)
    parser.add_argument("--clasifi_2_reg_flag", type=bool)
    parser.add_argument("--under_sampling_flag", type=bool)
    parser.add_argument("--mx_label_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--hidden_dropout_prob", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--beta", type=float)
    
    ## Set seed
    set_seeds()

    ## Static parameters
    config = parser.parse_args()
    config.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    ## Reset the Memory
    torch.cuda.empty_cache()

    ## Get transformer & tokenizer
    selection = Selection(config.model_name, config)
    model = selection.get()
    tokenizer = selection.unk_tokenizer_add(config)

    ## Get Data platform
    data_platform = DataPlatform(config, tokenizer)
    train_loader = data_platform.get_train_loader()
    val_loader = data_platform.get_val_loader()
    
    if config.kfold_flag:
        trainer = KFold(model, tokenizer, data_platform, config)
        trainer.train()
    else:
        ##############
        ## Training ##
        ##############
        trainer = Trainer(model, train_loader, val_loader, config, tokenizer)
        pearson_lst = [-0.5]
        for e in range(config.epoch):
            print("##########################################################")
            print("----------------------epoch {} start----------------------".format(e + 1))
            trainer.train(e, pearson_lst)
        
        print("#########################################")
        print("#########################################")
        print("--------------- Finished ----------------")
        test = Test(config, data_platform)
        test.test()
        test.make_submission_file()
        print("---------- Submission File OK ----------")
    
    
    