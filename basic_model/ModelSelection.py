from transformers import AutoTokenizer, AutoModel
from model import ModelBase, ModelFullClasifi
from Data import DataPlatform
import pandas as pd

class Selection():
    def __init__(self, model_name, config):
        
        self.config = config
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data = pd.read_csv(config.train_data_path)
        self.model = None
        
        self.small = ["klue/roberta-small", "beomi/KcELECTRA-small-v2022", "monologg/koelectra-small-v3-discriminator"]
        self.base = ["klue/roberta-base", "beomi/KcELECTRA-base", "monologg/koelectra-base-v3-discriminator", "beomi/kcbert-base", "jinmang2/kpfbert"]
        self.large = ["klue/roberta-large"]
        
        self.small_first_in = 256
        self.base_first_in = 768
        self.large_first_in = 1024
        
        ## Load Model
        if model_name in self.small:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag:
                self.model = ModelBase.Base(self.transformer, self.small_first_in, self.config.hidden_dropout_prob)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.small_first_in)
        elif model_name in self.base:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag:
                self.model = ModelBase.Base(self.transformer, self.base_first_in, self.config.hidden_dropout_prob)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.base_first_in)
        elif model_name in self.large:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag:
                self.model = ModelBase.Base(self.transformer, self.large_first_in, self.config.hidden_dropout_prob)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.large_first_in)
        
    ## UNK tokenizer add
    def unk_tokenizer_add(self, config):
        data_platform = DataPlatform(config, self.tokenizer)
        data_platform.concat_text(self.data)
        sentence = self.data['concat-text'].tolist()
        
        final_unk_lst = []
        
        for i in range(len(sentence)):
            token_lst = self.tokenizer.tokenize(sentence[i])
            for token in token_lst:
                if token == '[UNK]':
                    correct_sentence = sentence[i].replace(" ","")
                    unknowned_sentence = ' '.join(self.tokenizer.tokenize(sentence[i])).replace(' ##', '').strip().replace(" ","")
                    lst = list(unknowned_sentence.split("[UNK]"))
                    lst = ' '.join(lst).split()
                    for j in lst:
                        correct_sentence = correct_sentence.replace(j," ",1).strip()
                    unk_lst = list(correct_sentence.split(" "))
                    for unk_word in unk_lst:
                        if unk_word not in final_unk_lst:
                            final_unk_lst.append(unk_word)

        add_token_list = []

        for p in range(len(final_unk_lst)):
            unk_text = final_unk_lst[p]
            for q in range(len(unk_text)):
                token, token_text = self.tokenizer.tokenize(unk_text[q]), unk_text[q]
                if token == ['[UNK]']:
                    if q == 0:
                        add_token_list.append(token_text)
                        add_token_list.append('##' + token_text)
                    else:
                        add_token_list.append(token_text)
                        add_token_list.append('##' + token_text)
                        
        real_unk_token = list(set(add_token_list))
        add_token_num = self.tokenizer.add_tokens(real_unk_token)
        self.transformer.resize_token_embeddings(self.tokenizer.vocab_size + add_token_num)
        
        return self.tokenizer
            
    def get(self):
        return self.model