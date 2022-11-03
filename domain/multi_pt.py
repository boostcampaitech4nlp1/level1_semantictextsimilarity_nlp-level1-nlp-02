import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from model import ModelBase, ModelFullClasifi

class Prediction():
    '''''
    Model1 = 청원데이터
    Model2 = SLACK이랑 이런 데이터
    '''''
    
    def __init__(self, model1_path, model2_path, model1_name, model2_name, test_data_path, save_path):
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        self.save_path = save_path
        
        self.test_data = pd.read_csv(test_data_path)
        self.concat_text(self.test_data)
        self.test_text = self.test_data["concat-text"]
        
        self.transformer1 = AutoModel.from_pretrained(model1_name)
        self.tokenizer1 = AutoTokenizer.from_pretrained(model2_name)
        
        self.transformer2 = AutoModel.from_pretrained(model1_name)
        self.tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
        
        ## Here we need to add special token
        
        self.small = ["klue/roberta-small", "beomi/KcELECTRA-small", "monologg/koelectra-small-v3-discriminator"]
        self.base = ["klue/roberta-base", "beomi/KcELECTRA-base-v2022", "monologg/koelectra-base-v3-discriminator", "beomi/kcbert-base", "jinmang2/kpfbert"]
        self.large = ["klue/roberta-large"]
        
        self.small_first_in = 256
        self.base_first_in = 768
        self.large_first_in = 1024
        
        self.model1 = None
        self.model2 = None
        
        if model1_name in self.small:
            self.model1 = ModelBase.Base(self.transformer1, self.samll_first_in)
        elif model1_name in self.base:
            self.model1 = ModelBase.Base(self.transformer1, self.base_first_in)
        elif model1_name in self.large:
            self.model1 = ModelBase.Base(self.transformer1, self.large_first_in)
        
        if model2_name in self.small:
            self.model2 = ModelBase.Base(self.transformer2, self.samll_first_in)
        elif model2_name in self.base:
            self.model2 = ModelBase.Base(self.transformer2, self.base_first_in)
        elif model2_name in self.large:
            self.model2 = ModelBase.Base(self.transformer2, self.large_first_in)
        
        self.model1.load_state_dict(torch.load(model1_path))
        self.model2.load_state_dict(torch.load(model2_path))
        
        
    def concat_text(self, data):
        store = []
        for i in range(len(data)):
            sentence1 = data["sentence_1"][i]
            sentence2 = data["sentence_2"][i]

            concat_sentence = sentence1 + " [SEP] " + sentence2
            store.append(concat_sentence)
        
        data["concat-text"] = store
    
    def pred(self):
        self.test_score_store = []
        self.patition = ["patition-sampled", "patition-rtt"]
        self.slack_nsmc = ["nsmc-sampled", "nsmc-rtt", "slack-rtt", "slack-sampled"]
        
        for i in range(len(self.test_data)):
            now_test = self.test_text[i]
            if self.test_data["source"][i] in self.slack_nsmc:
                out = self.tokenizer1.encode_plus(
                    max_length=256,
                    truncation=True,
                    pad_to_max_length=True,
                    add_special_tokens=True,
                )
            else:
                out = self.tokenizer2.encode_plus(
                    max_length=256,
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
                if self.test_data["source"][i] == "nsmc-sampled" or self.test_data["source"][i] == "slack-rtt":
                    out_score, _ = self.model1(idz, attentions, token_types)
                else:
                    out_score, _ = self.model2(idz, attentions, token_types)
            
            self.test_score_store.append(round(out_score.cpu().numpy()[0][0], 1))
    
    def make_submission_file(self):
        self.test_data["target"] = self.test_score_store
        final_output = self.test_data.loc[:, ["id", "target"]]
        final_output.to_csv(self.save_path, index=False)