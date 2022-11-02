#!/bin/bash
python3 main.py \
--model1_path=./savedModel-koelectra-petition_only.pt \
--model2_path=./slack_nsmc_savedModel-1.pt \
--model1_name=monologg/koelectra-base-v3-discriminator \
--model2_name=beomi/KcELECTRA-base-v2022 \
--test_data_path=./test_space.csv \
--save_path=./tttemp-1.csv