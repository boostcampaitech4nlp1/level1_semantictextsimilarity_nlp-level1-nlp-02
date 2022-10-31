#!/bin/bash
python3 main.py \
--model_name=monologg/koelectra-base-v3-discriminator \
--batch_size=4 --epoch=10 \
--num_hidden_layer=2 --mx_token_size=256 \
 --kfold_flag=$False --early_stopping_flag=$False \
--only_clasifi_flag=True --only_reg_flag=$False \
--reg_plus_clasifi_flag=$False --clasifi_2_clasifi_flag=$False \
--clasifi_2_reg_flag=$False \
--lr=5e-6 --eps=1e-8 \
--save_path=../aistage/savedModel-1.pt --result_path=../aistage/sub-1.csv \
--train_data_path=../data/train.csv --val_data_path=../data/dev.csv --test_data_path=../data/test.csv