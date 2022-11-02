from transformers import AutoTokenizer, AutoModel
from model import ModelBase, ModelFullClasifi

class Selection():
    def __init__(self, model_name, config):
        
        self.config = config
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
        self.small = ["klue/roberta-small", "beomi/KcELECTRA-small", "monologg/koelectra-small-v3-discriminator"]
        self.base = ["klue/roberta-base", "beomi/KcELECTRA-base", "monologg/koelectra-base-v3-discriminator", "beomi/kcbert-base", "jinmang2/kpfbert"]
        self.large = ["klue/roberta-large"]
        
        self.small_first_in = 256
        self.base_first_in = 768
        self.large_first_in = 1024
        
        print(config)
        
        ## Load Model
        if model_name in self.small:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag:
                self.model = ModelBase.Base(self.transformer, self.small_first_in)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.small_first_in, self.config.hidden_dropout_prob)
        elif model_name in self.base:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag:
                self.model = ModelBase.Base(self.transformer, self.base_first_in)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.base_first_in, self.config.hidden_dropout_prob)
        elif model_name in self.large:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag:
                self.model = ModelBase.Base(self.transformer, self.large_first_in)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.large_first_in, self.config.hidden_dropout_prob)
            
    def get(self):
        return self.model, self.tokenizer