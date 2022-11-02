from transformers import AutoTokenizer, AutoModel
from model import ModelBase, ModelFullClasifi

class Selection():
    def __init__(self, model_name, config):
        
        self.config = config
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
        self.small = ["klue/roberta-small", "beomi/KcELECTRA-small-v2022", "monologg/koelectra-small-v3-discriminator"]
        self.base = ["klue/roberta-base", "beomi/KcELECTRA-base", "monologg/koelectra-base-v3-discriminator", "beomi/kcbert-base", "jinmang2/kpfbert"]
        self.large = ["klue/roberta-large"]
        
        self.small_first_in = 256
        self.base_first_in = 768
        self.large_first_in = 1024
        
        ## UNK tokenizer add
        if model_name in ["klue/roberta-small", "klue/roberta-base", "klue/roberta-large"]:
            self.add_token_num = self.tokenizer.add_tokens(['뎀', '혓', '##믓', '셧', '괸', '퐈', '##얔', '##ㅔ', '##됬', '##뼛', '##킵', 
                                                '##넼', '##쑈', '꺅', '##춋', '##꿎', '##왘', '##렜', '##돠', '##숀', '됫', '##롷', 
                                                '##멎', '##욥', '##ㅉ', '##힝', '챗', '빕', '꺠', '##꿉', '##챠', '좍', '쉑', '##쩝',
                                                '꺄', '돠', '##ㅑ', '빂', '##봣', '##쥴', '앖', '쫑', '##쐬', '챠', '뎃', '##늡', 
                                                '##댱', '##앳', '숀', '듄', '푀', '퉷', '##늣', '##홱', '앜', '##죵', '##낑', '##쨰',
                                                '##앜', '##ㅞ', '##웟', '뵛', '##쌰', 'ㅛ', '##딨', '##쨕', '눙', '뭍', '##탯', 
                                                '떄', '숴', '##횽', '##펐', '##셩', '##줜', '웤', '##옄', 'ㅓ', '좠', '##ᆢ', '##쎠',
                                                '콱', '##훃', '☼', '##웁', '끅', '쎴', '헿', '##쁩', '##닠', '##뎀', '##홥', '뜹', 
                                                '벴', '욯', '##핳', '쵝', '횐', '##찝', '킁', '##짘', '믕', '##헙', '##늼', '##됍', 
                                                '쵯', '##멱', '뗀', '##쟌', '뭏', '꼐', '껐', '##젬', '낑', '괞', '홱', '웩', '##넝',
                                                '##휏', '짦', '##ㅓ', '겜', '엊', '##ㅛ', '##잌', '솓', '##겆', '됬', '##밈', '홓', 
                                                '펭', '웟', '##꽈', '맬', '##넴', '##힣', '##힛', '큽', 'ㅊ', '뵐', '##뀰', '##윱', 
                                                '##듕', '뎁', '왘', '퀼', '뵌', '##샸', '뽈', '놋', '즁', '##쾃', '꿎', '##젼', '훅',
                                                '##떄', '됏', '쎈', '##낍', '홧', '##촤', '갬', '뽜', '##켤', '줸', '##셧', '##뇽', 
                                                '쯧', 'ㅍ', '앍', '##홐', '##췃', '뿨', '뼛', '##괸', '뇽', '죤', '즤', '뿅', '낟', 
                                                '봬', '##캅', '뙇', '멎', '##욬', '##씸', '찝', '칻', '쐬', '힛', '##ㅐ', '##럤', 
                                                '##끅', '##👌', '##즠', '땈', '##픕', '챕', '힝', '믓', '잴', '##낟', '햐', '뵀', 
                                                '##봬', '##튭', '##ㄳ', 'ㅉ', '옙', '퀵', '##쟝', '##돜', '##챗', '##홰', '밨', '봣',
                                                '쳬', '옄', '##껀', '굠', '멥', '##튱', '쑈', '얍', '욥', '##랖', '##푀', '##좍', 
                                                '핳', '앳', '##쯧', '솨', '힣', '켤', '졋', '##싀', '👌', '##쉑', '##킁', '뺄', 
                                                '##쉘', '##웻', '##숑', '##뀝', '욬', '##겜', '##롸', '뀰', '닼', '쩝', '왤', '훠', 
                                                '##넵', '넵', '꽈', '믱', '갭', '죵', '##탆', 'ㅑ', '##닼', '##굣', '｀', '##큩', 
                                                '쳇'])
            self.transformer.resize_token_embeddings(self.tokenizer.vocab_size + self.add_token_num)
            
        elif model_name in ["monologg/koelectra-small-v3-discriminator", "monologg/koelectra-base-v3-discriminator"]:
            self.add_token_num = self.tokenizer.add_tokens(['##믓', '퐈', '##얔', '##넼', '##춋', '##왘', '##렜', '##롷', '꺠', '좍', '빂', 
                                                '앖', '##늡', '##댱', '퉷', '##늣', '앜', '##쨰', '##앜', '##ㅞ', '뵛', '##쌰', 
                                                '##쨕', '##횽', '##줜', '웤', '##옄', '##ᆢ', '좠', '##쎠', '##훃', '☼', '끅', 
                                                '쎴', '##닠', '##홥', '벴', '욯', '##핳', '쵝', '횐', '##짘', '믕', '##늼', '##됍', 
                                                '쵯', '뭏', '괞', '##휏', '짦', '솓', '##겆', '홓', '##힣', '##뀰', '##윱', '##듕', 
                                                '왘', '##샸', '##쾃', '뽜', '줸', '앍', '##홐', '##췃', '뿨', '즤', '##욬', '칻', 
                                                '##럤', '##끅', '##👌', '##즠', '땈', '믓', '##튭', '##ㄳ', '##돜', '밨', '쳬', 
                                                '옄', '굠', '##튱', '##좍', '핳', '힣', '##싀', '👌', '##웻', '욬', '뀰', '닼', 
                                                '믱', '##탆', '##닼', '｀', '##큩'])
            self.transformer.resize_token_embeddings(self.tokenizer.vocab_size + self.add_token_num)
            
        elif model_name in ["beomi/KcELECTRA-base"]:
            self.add_token_num = self.tokenizer.add_tokens(['##쾃', '빂', '##ᆢ', '칻', '굠', '뿨', '믱', '##탆', '##훃', '☼', '##👌', '뵛', 
                                                            '👌', '##즠', '｀', '##큩'])
            self.transformer.resize_token_embeddings(self.tokenizer.vocab_size + self.add_token_num)
            
        elif model_name in ["beomi/KcELECTRA-small-v2022"]:
            self.add_token_num = self.tokenizer.add_tokens(['믱', '##홐', '굠', '뿨', '##웻', '##탆', '뵀', '##즠', '칻', '좠', '##쨕', '빂', '뵛', 
                                                            '멥', '쵯', '##훃', '##럤', '##큩', '☼', '벴', '뀰', '욯', '##뀰', '##윱', '##쾃'])
            self.transformer.resize_token_embeddings(self.tokenizer.vocab_size + self.add_token_num)
            
        elif model_name in ["beomi/kcbert-base"]:
            self.add_token_num = self.tokenizer.add_tokens(['ㅏ', 'ㅓ', '##쾃', '##ᆢ', '>', '굠', '“', '멥', '’', '##훃', '☼', '##ㅔ', 
                                                        'ㅡ', 'ㅣ', '…', '##홐', '뿨', '##<', '##ㅣ', '##ᆞ', '<', '쵯', '##웻', '##ㅑ', 
                                                        '빂', '##ㅜ', '##ㅡ', '‥', 'ㅠ', 'ㅤ', '칻', '##ㅐ', '##즠', '&', '뀰', '##ㅓ', 
                                                        '##ㅛ', 'ㅜ', '”', '##뀰', '##…', '믱', '##ㅠ', '‘', '##ㅞ', '##탆', 'ㅑ', '##>', 
                                                        '뵛', 'ㅛ', '｀', '##ㅏ', '##큩'])
            self.transformer.resize_token_embeddings(self.tokenizer.vocab_size + self.add_token_num)
            
        elif model_name in ["jinmang2/kpfbert"]:
            self.add_token_num = self.tokenizer.add_tokens(['##믓', '괸', '퐈', '##얔', '##쑈', '##춋', '##왘', '##렜', '##돠', '됫', '##롷', 
                                                        '꺠', '좍', '돠', '빂', '앖', '##늡', '##댱', '듄', '퉷', '##늣', '##쨰', '##ㅞ', 
                                                        '뵛', '##쌰', '##쨕', '##횽', '##줜', '웤', '##옄', '##ᆢ', '좠', '##쎠', '##훃', 
                                                        '☼', '끅', '쎴', '##닠', '##홥', '뜹', '벴', '욯', '쵝', '횐', '##짘', '믕', '##늼',
                                                        '##됍', '쵯', '뭏', '꼐', '괞', '##휏', '짦', '솓', '##겆', '홓', '##힣', '##뀰', 
                                                        '##윱', '##듕', '왘', '##샸', '##쾃', '뽜', '줸', '앍', '##홐', '##췃', '뿨', '##괸',
                                                        '즤', '칻', '##럤', '##끅', '##즠', '땈', '믓', '뵀', '##튭', '##ㄳ', '##돜', '밨',
                                                        '쳬', '옄', '굠', '멥', '##튱', '쑈', '##좍', '솨', '힣', '##싀', '##웻', '##뀝', 
                                                        '##롸', '뀰', '믱', '##탆', '｀', '##큩'])
            self.transformer.resize_token_embeddings(self.tokenizer.vocab_size + self.add_token_num)
        
        ## Load Model
        if model_name in self.small:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag or self.config.reg_plus_multi_clasifi_flag:
                self.model = ModelBase.Base(self.transformer, self.small_first_in, self.config.hidden_dropout_prob)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.small_first_in)
        elif model_name in self.base:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag or self.config.reg_plus_multi_clasifi_flag:
                self.model = ModelBase.Base(self.transformer, self.base_first_in, self.config.hidden_dropout_prob)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.base_first_in)
        elif model_name in self.large:
            if self.config.reg_plus_clasifi_flag or self.config.only_reg_flag or self.config.reg_plus_multi_clasifi_flag:
                self.model = ModelBase.Base(self.transformer, self.large_first_in, self.config.hidden_dropout_prob)
            if self.config.only_clasifi_flag:
                self.model = ModelFullClasifi.FullClasifi(self.transformer, self.large_first_in)
            
    def get(self):
        return self.model, self.tokenizer