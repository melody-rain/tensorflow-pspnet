#!/usr/bin/env bash

python test_segmentation.py \
--checkpoint_path train/pspnet \
--model_name pspnet_v1_50 \
--image_list IMG_LIST \
--data_root DATA_ROOT

