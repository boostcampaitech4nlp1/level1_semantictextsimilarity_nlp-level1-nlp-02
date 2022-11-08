import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model_selection import Selection
from data import DataPlatform
from trainer import Trainer
from test import Test
import copy

class KFold():
    def __init__(self, model, tokenizer, data_platform, config):
        ## Model & Tokenizer
        self.inital_model = copy.deepcopy(model)
        self.model = model
        self.tokenizer = tokenizer
        
        ## Data Platform & Data
        self.data_platform = data_platform
        self.train_data = self.data_platform.get_train_data()
        self.val_data = self.data_platform.get_val_data()
        
        ## Config
        self.config = config
        
        ## Kfold file store
        self.file_store = []
        
        ## Seperate data
        self.seperate_data(self.config.kfold_k)
        
        ## Initial save model name and csv file name
        self.model_save_path = self.config.save_path[:-3]
        self.result_save_path = self.config.result_path[:-4]
        
    def seperate_data(self, k):
        ## Get labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self.train_data["label"])
        labels = label_encoder.classes_
        length = len(labels)
        
        temp_labels = self.train_data["label"]
        counter = temp_labels.value_counts()
        
        data_store = [[] for i in range(k)]
        
        for i in range(length):
            ## Random shuffle data
            now_data = self.train_data.loc[self.train_data["label"] == labels[i]]
            sampling = now_data.sample(
                n=len(now_data),
                random_state=random.randrange(1, 10000),
                replace=False,
                )
            
            ## Seperate the dataset
            block_size = int(len(now_data) / k)
            for j in range(k):
                temp = sampling.iloc[j*block_size:(j+1)*block_size]
                data_store[j].append(temp)
                        
        ## Concat the files
        for i in range(k):
            now = data_store[i][0]
            for j in range(1, len(data_store[i])):
                now = pd.concat([now, data_store[i][j]], axis=0)
            self.file_store.append(now)

    def fold_file_maker(self, num):
        now = pd.DataFrame()
        for i in range(len(self.file_store)):
            if i == num: continue
            if now.empty: now = self.file_store[i]
            else:
                now = pd.concat([now, self.file_store[i]], axis=0)
        return now
    
    def train(self):
        
        pearson_list = []
        
        for i in range(len(self.file_store)):
            ## Make data loader
            train_data = self.fold_file_maker(i)
            self.data_platform.train_data = train_data
            self.data_platform.make_reg_clasi_data_loader()

            ## Get data loader
            train_loader = self.data_platform.get_train_loader()
            val_loader = self.data_platform.get_val_loader()
            
            ## Change save path name
            self.config.save_path = self.model_save_path + str(i+1) + ".pt"
            self.config.result_path = self.result_save_path + str(i+1) + ".csv"
                        
            ## Training
            trainer = Trainer(self.model, train_loader, val_loader, self.config, self.tokenizer)
            pearson_lst = [-0.5]
            for e in range(self.config.epoch):
                print("##########################################################")
                print("----------------------epoch {} start----------------------".format(e + 1))
                trainer.train(e, pearson_lst)
            
            ## Get mx pearson
            pearson_list.append(trainer.mx_pearson)
            
            ## Test
            print("#########################################")
            print("#########################################")
            print("--------------- Finished ----------------")
            test = Test(self.config, self.data_platform)
            test.test()
            test.make_submission_file()
            print("---------- Submission File OK ----------")

            self.model = copy.deepcopy(self.inital_model)
            
        ## Write the result
        pearson_result = open(self.model_save_path+".txt", "w")
        pearson_result.write(str(pearson_list))
        