#!/bin/bash

EVAL_DIR=./eval/pspnet
DATASET_DIR=YOUR_DATASET_PATH
CHECKPOINT_PATH=./train/pspnet

python eval_semantic_segmentation.py \
--eval_dir=${EVAL_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=YOUR_DATASET_NAME \
--dataset_split_name=val \
--model_name=pspnet_v1_101 \
--checkpoint_path=${CHECKPOINT_PATH} \
--eval_image_size=473 \
--batch_size=8
