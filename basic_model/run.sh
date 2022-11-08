#!/bin/bash
python3 main.py \
--model_name=beomi/KcELECTRA-base \
--batch_size=4 --epoch=10 \
--num_hidden_layer=2 --mx_token_size=128 \
--kfold_flag=True --kfold_k=5 \
--early_stopping_flag=$False \
--only_clasifi_flag=$False --only_reg_flag=$False \
--reg_plus_clasifi_flag=True \
--reg_plus_multi_clasifi_flag=$False \
--clasifi_2_clasifi_flag=$False \
--clasifi_2_reg_flag=$False \
--under_sampling_flag=True --mx_label_size=450 \
--lr=5e-6 --eps=1e-8 \
--hidden_dropout_prob=0.1 --weight_decay=0.01 --beta=0.6 \
--save_path=../save/kccccc_kfold.pt --result_path=../save/kccccc_kfold.csv \
--train_data_path=../data/train_space.csv --val_data_path=../data/dev_space.csv --test_data_path=../data/test_space.csv


#!/bin/bash
python3 main.py \
--model_name=jinmang2/kpfbert \
--batch_size=4 --epoch=10 \
--num_hidden_layer=2 --mx_token_size=128 \
--kfold_flag=True --kfold_k=5 \
--early_stopping_flag=$False \
--only_clasifi_flag=$False --only_reg_flag=$False \
--reg_plus_clasifi_flag=True \
--reg_plus_multi_clasifi_flag=$False \
--clasifi_2_clasifi_flag=$False \
--clasifi_2_reg_flag=$False \
--under_sampling_flag=True --mx_label_size=450 \
--lr=5e-6 --eps=1e-8 \
--hidden_dropout_prob=0.1 --weight_decay=0.01 --beta=0.6 \
--save_path=../save/kpfbert_kfold.pt --result_path=../save/kpfbert_kfold.csv \
--train_data_path=../data/train_space.csv --val_data_path=../data/dev_space.csv --test_data_path=../data/test_space.csv