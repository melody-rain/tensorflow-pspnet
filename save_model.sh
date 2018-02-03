#!/usr/bin/env bash

python save_model.py \
--checkpoint_path train/pspnet \
--model_name pspnet_v1_101 \
--output_dir models \
--output_filename pspnet_v1_101.pb \
