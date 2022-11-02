
import torch
from Data import DataPlatform
from ModelSelection import Selection

class Test():
    def __init__(self, config, data_platform):
        ## Device
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        ## Infomation
        self.config = config
        self.data_platform = data_platform
        
        ## Test Data
        self.test_data = self.data_platform.test_data
        self.test_text = self.test_data["concat-text"]
        
        ## Get model and tokenizer
        selection = Selection(self.config.model_name, self.config)
        self.model = selection.get()
        self.tokenizer = selection.unk_tokenizer_add(config)
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.to(self.device)
        
        ## Store
        self.test_score_store = []
        
    def test(self):
        
        for i in range(len(self.test_text)):
            now_text = self.test_text[i]
            
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
            #self.test_score_store.append(round(out_score.cpu().numpy()[0][0], 1))
            self.test_score_store.append(out_score.cpu().numpy[0][0])
            
    def make_submission_file(self):
        self.test_data["target"] = self.test_score_store
        final_output = self.test_data.loc[:, ["id", "target"]]
        final_output.to_csv(self.config.result_path, index=False)