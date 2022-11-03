#!/bin/bash
python3 main.py \
--model1_path=../../saved/savedModel-koelectra-petition_only.pt \
--model2_path=../../saved/slack_nsmc_savedModel-1.pt \
--model1_name=monologg/koelectra-base-v3-discriminator \
--model2_name=beomi/KcELECTRA-base-v2022 \
--test_data_path=../../data/test_space.csv \
--save_path=../../saved/domain_model.csv