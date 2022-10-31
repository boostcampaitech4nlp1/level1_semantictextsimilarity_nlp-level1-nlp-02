
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

########################
## Main Data Platform ##
########################

class DataPlatform():
    def __init__(self, config, tokenizer):
        ## Save tokenizer & config
        self.tokenizer = tokenizer
        self.config = config

        ## Get Data
        self.train_data = pd.read_csv(config.train_data_path)
        self.val_data = pd.read_csv(config.val_data_path)
        self.test_data = pd.read_csv(config.test_data_path)
        self.over_sampling_data = None

        ## Concat two sentence
        self.concat_text(self.train_data)
        self.concat_text(self.val_data)
        self.concat_text(self.test_data)

        ## Data Loader
        self.train_loader = None
        self.val_loader = None
        
        ## Get Label & Label encoding
        if self.config.only_reg_flag or self.config.reg_plus_clasifi_flag:
            self.make_reg_clasi_data_loader()
        else:
            self.labels = None
            self.label_encoding_full()
            self.make_full_clasi_data_loader()
            

    def concat_text(self, data):
        store = []
        for i in range(len(data)):
            sentence1 = data["sentence_1"][i]
            sentence2 = data["sentence_2"][i]

            concat_sentence = sentence1 + " [SEP] " + sentence2
            store.append(concat_sentence)
        
        data["concat-text"] = store

    def label_encoding_full(self):
        label_encoder = LabelEncoder()
        label_encoder.fit(self.train_data["label"])
        self.train_data["full-class"] = label_encoder.transform(self.train_data["label"].values)
        self.val_data["full-class"] = label_encoder.transform(self.val_data["label"].values)
        self.labels = label_encoder.classes_
        
    def label_encoding_part(self):
        self.train_data["part-class"] = -1
        self.val_data["part-class"] = -1
        return
    
    def make_reg_clasi_data_loader(self):
        '''''
        Dataset will return score and binary class
        It will be used for regression and classification
        '''''
        train = DataRegClasi(self.train_data, self.tokenizer, self.config)
        val = DataRegClasi(self.val_data, self.tokenizer, self.config)

        self.train_loader = DataLoader(
            train,
            shuffle = True,
            batch_size = self.config.batch_size,
        )
        self.val_loader = DataLoader(
            val,
            shuffle = True,
            batch_size = self.config.batch_size,
        )
    
    def make_full_clasi_data_loader(self):
        '''''
        Dataset will return multi class and binary class
        It will be used for only classification
        '''''
        train = DataFullClasi(self.train_data, self.tokenizer, self.config)
        val = DataFullClasi(self.val_data, self.tokenizer, self.config)
        
        self.train_loader = DataLoader(
            train,
            shuffle = True,
            batch_size = self.config.batch_size,
        )
        self.val_loader = DataLoader(
            val,
            shuffle = True,
            batch_size = self.config.batch_size,
        )
    
    def make_clasi_part_data_loader(self):
        return

    def under_sampling(self):
        return

    def over_sampling(self):
        return

    def get_train_loader(self): return self.train_loader
    def get_val_loader(self): return self.val_loader
    def get_label(self): return self.labels
    def get_train_data(self): return self.train_data
    def get_val_data(self): return self.val_data
    def get_test_data(self): return self.test_data


#############
## Dataset ##
###########################################
# * Different Method, Different Dataset * #
###########################################

class DataRegClasi(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __getitem__(self, idx):
        out = self.tokenizer.encode_plus(
            list(self.data["concat-text"])[idx],
            return_tensors="pt",
            max_length=self.config.mx_token_size,
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        input_ids = out["input_ids"][0]
        attention_mask = out["attention_mask"][0]
        token_type_ids = out["token_type_ids"][0]

        ## input_ids, attention_mask, score, T/F
        return input_ids, attention_mask, token_type_ids, list(self.data["label"])[idx], list(self.data["binary-label"])[idx]

    def __len__(self):
        return len(self.data)

class DataFullClasi(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
    
    def __getitem__(self, idx):
        out = self.tokenizer.encode_plus(
            list(self.data["concat-text"])[idx],
            return_tensors="pt",
            max_length=self.config.mx_token_size,
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        input_ids = out["input_ids"][0]
        attention_mask = out["attention_mask"][0]
        token_type_ids = out["token_type_ids"][0]
        
        return input_ids, attention_mask, token_type_ids, list(self.data["full-class"])[idx], list(self.data["binary-label"])[idx]
        
    def __len__(self):
        return len(self.data)