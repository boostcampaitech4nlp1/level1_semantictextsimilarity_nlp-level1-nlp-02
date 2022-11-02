import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from multi_pt import Prediction


if __name__ == "__main__":
    ## Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_path", type=str)
    parser.add_argument("--model2_path", type=str)
    parser.add_argument("--model1_name", type=str)
    parser.add_argument("--model2_name", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--save_path", type=str)
    config = parser.parse_args()
    
    prediction = Prediction(
        config.model1_path,
        config.model2_path,
        config.model1_name,
        config.model2_name,
        config.test_data_path,
        config.save_path,
    )
    
    ## Reset the Memory
    torch.cuda.empty_cache()
    
    prediction.pred()
    prediction.make_submission_file()
    
    
    
    