#!/bin/bash

EVAL_DIR=./eval/pspnet
DATASET_DIR=YOUR_DATASET_PATH
CHECKPOINT_PATH=./train/pspnet

python eval_semantic_segmentation.py \
--eval_dir=${EVAL_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=ade20k \
--dataset_split_name=validation \
--model_name=pspnet_v1_50 \
--checkpoint_path=${CHECKPOINT_PATH} \
--eval_image_size=473 \
--batch_size=10 \
--freeze_bn \