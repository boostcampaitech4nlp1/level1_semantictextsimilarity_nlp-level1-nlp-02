## pip install --ignore-installed PyYAML
## pip install -r requirements.txt

import argparse
import torch.nn as nn
import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from omegaconf import OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Dataset(torch.utils.data.Dataset): ## __getitem__, __len__ 꼭 필요하다. len만큼 가져옴
    def __init__(self, inputs, targets=[], targets_2 = []):
        self.inputs = inputs
        self.targets = targets
        self.targets_2 = targets_2
        
    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx]), torch.tensor(self.targets_2[idx])
        
    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)
    
## dataset에 접근하기 위한 module, literally data loader
class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.target_columns_2 = ['binary-label'] ## target columns를 binary label도 사용
        self.delete_columns = ['id'] ## 'id'는 쓸모 x
        self.text_columns = ['sentence_1', 'sentence_2']
        
        
        
    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data
## 삭제할만한 columns를 삭제하고, data를 tokenizing하고 inputs, [targets, targets_2] 반환. literally preprocessing
    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)
        # 타겟 데이터가 없으면 빈 배열을 리턴합니다. ## 비어있다면 오류를 호출할테니 except를 실행
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        try:
            targets_2 = data[self.target_columns_2].values.tolist()
        except:
            targets_2 = []
        # 텍스트 데이터를 전처리합니다.
        
        inputs = self.tokenizing(data)
        
        return inputs, targets, targets_2
    
## stage에 따라 dataset을 준비하는 setup, trainer.fit(), trainer.test() 이전에 작동됨.
    def setup(self, stage='fit'): ## stage == 'fit'이냐 아니면 "test"냐. 참고로 stage always be one of {fit,validate,test,predict}
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)
            
            # 학습데이터 준비
            train_inputs, train_targets, train_targets_2 = self.preprocessing(train_data)
            
            # 검증데이터 준비
            val_inputs, val_targets, val_targets_2 = self.preprocessing(val_data)
            
            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets, train_targets_2)
            self.val_dataset = Dataset(val_inputs, val_targets, val_targets_2)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets, test_targets_2 = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets, test_targets_2)
            
            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets, predict_targets_2 = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [], [])
## 각 단계에 맞는 dataloader 함수 구현.
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle) ## num_workers 추가 bottleneck방지
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size) ## PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    
class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters() ## call this to save (model_name, lr, bce) to the checkpoint
        
        self.model_name = config.model.model_name
        self.lr = config.train.learning_rate
        
        self.weight_decay = config.train.weight_decay
        self.eps = config.train.eps
        # 사용할 모델을 호출합니다. ## AutoModelForSequenceClassification을 꼭 쓸 필요없다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1)
        # Loss 계산을 위해 사용될 MSELoss를 호출합니다.
        self.regression_loss_func = torch.nn.MSELoss()
        self.binary_loss_func = torch.nn.BCELoss()
        
        self.sigmoid = nn.Sigmoid()
        
## pre_trained model에 data를 입력하고 output을 반환하는 코드이다.
    def forward(self, x):
        y = self.plm(x)['logits']
        y_bi = self.sigmoid(self.plm(x)['logits'])
        return y, y_bi
## step code는 전부 학습 code이다. pytorch lightning의 문법
    def training_step(self, batch, batch_idx):
        inputs, target, target_bi = batch # (16, 512), (16, 1), (16, 1)
        # print(target)
        # print(target_bi)
        logits, logits_bi = self(inputs)
        loss_regression = self.regression_loss_func(logits, target.float())
        loss_classification = self.binary_loss_func(logits_bi, target_bi.float())
        loss = 0.65 * loss_regression + 0.35 * loss_classification
        self.log("train_loss", loss) ## train_loss라는 이름으로 log를 남기자. wandB에 자동으로 남게됨
        return loss
    
## validation_step
    def validation_step(self, batch, batch_idx):
        inputs, target, target_bi = batch
        logits, logits_bi = self(inputs)
        
        loss_regression = self.regression_loss_func(logits, target.float())
        loss_classification = self.binary_loss_func(logits_bi, target_bi.float())
        loss = 0.65 * loss_regression + 0.35 * loss_classification
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), target.squeeze()))
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, target, target_bi = batch
        logits, logits_bi = self(inputs)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), target.squeeze()))
        
    def predict_step(self, batch, batch_idx):
        inputs = batch
        logits, logits_bi = self(inputs)
        return logits.squeeze()
## optimizer setting. 여기서는 AdamW 사용. optimizer와 LR scheduler를 선언하는 함수이다.
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay) ## weight_decay 추가
        return optimizer
    
if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    ## parser에 익숙해지라는 뜻도 있지만, 여기서 hyperparmeter들을 바꿔보면서 최적화 할 수 있게 만든 것도 있다.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    # wandb logger 설정
    wandb.login()
    wandb_logger = WandbLogger(name='gustn9609', project='nlp2조_baseline_klue_roberta_small')
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(cfg.model.model_name, cfg.train.batch_size, cfg.data.shuffle, cfg.path.train_path, cfg.path.dev_path,
                            cfg.path.test_path, cfg.path.predict_path)
    model = Model(cfg)
    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, logger=wandb_logger, log_every_n_steps=cfg.train.logging_step) ##logger추가
    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    # 학습이 완료된 모델을 저장합니다. ## 'model.pt'라는 file에 저장한다.
    torch.save(model, f'{cfg.model.saved_name}.pt')