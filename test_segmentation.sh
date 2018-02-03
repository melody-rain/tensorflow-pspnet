#!/usr/bin/env bash

python test_segmentation.py \
--checkpoint_path train/pspnet \
--model_name pspnet_v1_101 \
--image PATH_TO_TEST_IMAGE \
