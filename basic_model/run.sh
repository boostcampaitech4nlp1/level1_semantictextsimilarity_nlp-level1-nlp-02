#!/bin/bash
python3 main.py \
--model_name=monologg/koelectra-base-v3-discriminator \
--batch_size=4 --epoch=30 \
--num_hidden_layer=2 --mx_token_size=128 \
--kfold_flag=$False --early_stopping_flag=$False \
--only_clasifi_flag=$False --only_reg_flag=$False \
--reg_plus_clasifi_flag=True --clasifi_2_clasifi_flag=$False \
--clasifi_2_reg_flag=$False \
--lr=5e-6 --eps=1e-8 \
--save_path=savedModel-3-realover.pt --result_path=sub-realover.csv \
--train_data_path=/opt/ml/code/real_over.csv --val_data_path=/opt/ml/code/dev_space.csv --test_data_path=/opt/ml/code/test_space.csv