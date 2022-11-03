import datetime
import numpy as np
import pandas as pd
import torch
import warnings
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from utils.Earlystopping import EarlyStopping
import sys
warnings.filterwarnings("ignore")

#############
## Trainer ## 
#############
class Trainer():
    def __init__(self, model, train_loader, val_loader, config, tokenizer):
        ## Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Config
        self.config = config

        ## Model
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        ## Data Loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        ## Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = self.config.lr,
            eps = self.config.eps,
            weight_decay = self.config.weight_decay
        )

        ## Loss Function
        self.regression_loss_fn = nn.MSELoss().to(self.device)
        self.bi_classification_loss_fn = nn.BCELoss().to(self.device)
        self.multi_classification_loss_fn = nn.CrossEntropyLoss().to(self.device)

        self.early_stopping = EarlyStopping(path = config.save_path, patience = 7, verbose = True)
        
        ## beta
        self.beta = self.config.beta

    def train(self,epoch,f):
        batch_count = 0
        train_loss_store = []
        pearson_lst = []
        for idz, attentions, token_types, score, bi_or_multi_class in self.train_loader:
            ## Load to cpu or gpu
            idz = idz.to(self.device)
            attentions = attentions.to(self.device)
            token_types = token_types.to(self.device)
            score = score.to(self.device)
            bi_or_multi_class = bi_or_multi_class.to(self.device)

            ## Predict with data
            out_score, out_bi_class, out_multi_class = self.model(idz, attentions, token_types)
            out_score = out_score.squeeze()
            out_bi_class = out_bi_class.squeeze()
            out_multi_class = out_multi_class.squeeze() ## [,,6]

            ## Train according to the method
            ## Caculate the loss
            if self.config.reg_plus_clasifi_flag:
                try:
                    loss = self.reg_plus_clasifi(out_score, out_bi_class, score, bi_or_multi_class)
                except:
                    continue
            elif self.config.reg_plus_multi_clasifi_flag:
                try:
                    loss = self.reg_plus_multi_clasifi(out_score, out_multi_class, score, bi_or_multi_class)
                except:
                    continue
            elif self.config.only_reg_flag:
                loss = self.only_reg(out_score, score)
            elif self.config.only_clasifi_flag:
                ## In this part, out_score = out_multi_class, score = multi_class
                loss = self.only_clasifi(out_score, out_bi_class, score, bi_or_multi_class)
                
            ## Store loss
            train_loss_store.append(loss)

            ## Optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ## +1 Batch count
            batch_count += 1
            
            ## Evaluation step
            if not batch_count % 400: 
                eval_loss = self.eval()
                pearson = self.pearson_score()
                print("Batch {} over".format(batch_count * self.config.batch_size))
                print("@@@@@@ Now evaluation loss : {} @@@@@@@".format(eval_loss))
                pearson_lst.append(pearson)
                print(pearson,file = f)
                print("!!!!!! Now pearson score : {} !!!!!!!".format(pearson))
                # self.save() # early stopping 코드에서 현재 최고 모델을 저장
                if epoch >= 5 and self.early_stopping.early_stop:
                    print("Early stopping")
                    self.early_stopping.flag = True
                    break

        print(f"{epoch}'s max pearson is {max(pearson_lst)}",file = f)

    def eval(self):
        total_loss = 0
        total_count = 0

        total_loss = 0
        total_count = 0
        
        store = []
        
        for idz, attentions, token_types, score, bi_or_multi_class in self.val_loader:
            ## Load to cpu or gpu
            idz = idz.to(self.device)
            attentions = attentions.to(self.device)
            token_types = token_types.to(self.device)
            score = score.to(self.device)
            bi_or_multi_class = bi_or_multi_class.type(torch.LongTensor).to(self.device)

            ## Predict with data & without gradient calculation
            ## !!! if you want to implement early stopping rewrite this code (without no_grad)
            out_score, out_bi_class, out_multi_class = self.model(idz, attentions, token_types)
            out_score = out_score.squeeze()
            out_multi_class = out_multi_class.squeeze()
            
            ## Calculate the loss
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag:
                loss = self.regression_loss_fn(out_score, score).detach()
            elif self.config.reg_plus_multi_clasifi_flag:
                loss = self.reg_plus_multi_clasifi(out_score, out_multi_class, score, bi_or_multi_class).detach()
            elif self.config.only_clasifi_flag:
                loss = self.multi_classification_loss_fn(out_score, score).detach()
            
            ## Store the out_score
            store.append(out_score.detach())
            
            ## Sum of Loss
            total_loss += loss
            
            ## Data size
            total_count += self.config.batch_size
            
        self.early_stopping(total_loss/total_count,self.model)
            
        return total_loss / total_count

    def pearson_score(self):
        ## Get data
        val_data = pd.read_csv(self.config.val_data_path)
        self.concat_text(val_data)
        val_text = val_data["concat-text"]
        
        store_result = []
        
        for i in range(len(val_data)):
            now_text = val_text[i]
            
            out = self.tokenizer.encode_plus(
                now_text,
                max_length=self.config.mx_token_size,
                truncation=True,
                pad_to_max_length=True,
                add_special_tokens=True,
            )

            idz = [out["input_ids"]]
            attentions = [out["attention_mask"]]
            token_types = [out["token_type_ids"]]

            idz = torch.tensor(idz).to(self.device)
            attentions = torch.tensor(attentions).to(self.device)
            token_types = torch.tensor(token_types).to(self.device)
            
            with torch.no_grad():
                if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag or self.config.reg_plus_multi_clasifi_flag:
                    out_score, _, _ = self.model(idz, attentions, token_types)
                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("We don't support classification task!")
                    break
            store_result.append(round(out_score.cpu().numpy()[0][0], 1))
        
        ## Get pearson score
        ground_truth = val_data["label"].to_numpy()
        
        return np.corrcoef(store_result, ground_truth)[0, 1]
    
    def reg_plus_clasifi(self, out_score, out_bi_class, score, bi_or_multi_class):
        loss_regression = self.regression_loss_fn(out_score.float(), score.float())
        loss_classification = self.bi_classification_loss_fn(out_bi_class.float(), bi_or_multi_class.float())
        loss = self.beta * loss_regression + (1-self.beta) * loss_classification
        return loss
    
    def reg_plus_multi_clasifi(self, out_score, out_multi_class, score, bi_or_multi_class):
        loss_regression = self.regression_loss_fn(out_score.float(), score.float())
        loss_multi_classification = self.multi_classification_loss_fn(out_multi_class, bi_or_multi_class)
        loss = self.beta * loss_regression + (1-self.beta) * loss_multi_classification
        return loss
    
    def only_reg(self, out_score, score):
        loss_regression = self.regression_loss_fn(out_score.float(), score.float())
        return loss_regression
    
    def only_clasifi(self, out_multi_class, out_bi_class, multi_class, bi_or_multi_class):
        loss_multi_classification = self.multi_classification_loss_fn(out_multi_class.float(), multi_class)
        loss_classification = self.bi_classification_loss_fn(out_bi_class.float(), bi_or_multi_class.float())
        loss = 0.7 * loss_multi_classification# + 0.3 * loss_classification
        return loss
    
    def concat_text(self, data):
        store = []
        for i in range(len(data)):
            sentence1 = data["sentence_1"][i]
            sentence2 = data["sentence_2"][i]

            concat_sentence = sentence1 + " [SEP] " + sentence2
            store.append(concat_sentence)
        
        data["concat-text"] = store
    
    def save(self):
        torch.save(self.model.state_dict(), self.config.save_path)