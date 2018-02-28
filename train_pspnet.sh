#!/bin/bash

# CHECKPOINT_PATH=./train/pretrained/pspnet_v1_50.ckpt
# CHECKPOINT_EXCLUDE_SCOPES=global_step:0,pspnet_v1_50/pyramid_pool_module,pspnet_v1_50/fc1,pspnet_v1_50/logits

TRAIN_DIR=./train/pspnet
DATASET_DIR=PATH_TO_DATASET
CHECKPOINT_PATH=PSPNET_MODEL_PATH
CHECKPOINT_EXCLUDE_SCOPES=pspnet_v1_101/aux_logits,pspnet_v1_101/logits # to train new dataset

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_semantic_segmentation.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --dataset_name=ade20k \  # change to your dataset name
  --dataset_split_name=train \
  --model_name=pspnet_v1_101 \
  --optimizer=momentum \
  --weight_decay=0.0001 \
  --max_number_of_steps=150000 \
  --train_image_size=473 \
  --batch_size=2 \
  --checkpoint_exclude_scopes=${CHECKPOINT_EXCLUDE_SCOPES} \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=20 \
  --learning_rate=0.0005 \
  --end_learning_rate=0.000005 \
  --learning_rate_decay_type=polynomial \
  --learning_rate_decay_factor=0.99 \
  --num_clones=4 \
#--freeze_bn \
