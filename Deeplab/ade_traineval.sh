#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run 1 train + eval on ADE20k as explained in 
# g3doc/ade20k.md with the purpose on compairing with a similar train +
# eval experiment using SPADE generated images
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./ade_traineval.sh
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo $PYTHONPATH

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# # Run model_test first to make sure the PYTHONPATH is correctly set.
# python "${WORK_DIR}"/model_test.py -v
# Set datasets folder 
DATASET_DIR="datasets/ADE20K"

# Set up the working directories.
ADE_FOLDER="ADEChallengeData2016"
EXP_FOLDER="exp/train_on_trainval"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${ADE_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

echo "-------------------"
echo "Created ADE folders"
echo "Init folder: ${INIT_FOLDER}"
echo "Train log dir: ${TRAIN_LOGDIR}"
echo "Eval log dir: ${EVAL_LOGDIR}"
echo "Vis log dir: ${VIS_LOGDIR}"
echo "Export folder: ${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

ADE_DATASET="${WORK_DIR}/${DATASET_DIR}/tfrecord"

# check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# export CUDA_VISIBLE_DEVICES in order to select which of our GPUs 
# must be used. It appears that Deeplab does the indexing by calling
# 0 the internal GPU, while 1 and 2 Pascal and GeForce respectively.
export CUDA_VISIBLE_DEVICES="1"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"

# Notes on train.py:
# - number of iteration is pretty higher than the one we used (150K  
#   againt usual 20K or 30K). Be careful!
# - if you want to fine tune the BatchNorm layers, user larger batch size
#   (bs>12) and set fine_tune_batch_norm = True. WARNING: this is quite 
#   costly, so don't use it with limited resources at hand
# - min_resize_value and max_resize_value should be fine tuned in order 
#   to get better results
# - resize_factor should be equal to output_stride
# - if output_stride==8, use atrous_rates [12,24,36]
# - if you don't want to use the decoder struture, just skip the flag
#   called decoder_output_stride
DATASET="ade20k"
NUM_ITERATIONS=30000
TRAIN_SPLIT="train"
EVAL_SPLIT="val"
BATCH_SIZE=8 # default
FINETUNE_BN="False"
CROP_SIZE=513
MAX_RESIZE=1024
ADE_CLASSES=151

echo "----------------------------"
echo "Starting training. Options:"
echo "Model variant: xception_65"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Dataset: ${DATASET}"
echo "Dataset split: ${TRAIN_SPLIT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Finetuning the Batch Norm layers: ${FINETUNE_BN}"
echo "Atrous rates: [6, 12, 18]"
echo "Output stride: 16"
echo "Resize factor: 16"
echo "Crop size: ${CROP_SIZE}"

# python "${WORK_DIR}"/train.py \
#   --logtostderr \
#   --training_number_of_steps="${NUM_ITERATIONS}" \
#   --train_split="${TRAIN_SPLIT}" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --train_crop_size="${CROP_SIZE},${CROP_SIZE}" \
#   --train_batch_size="${BATCH_SIZE}" \
#   --fine_tune_batch_norm="${FINETUNE_BN}" \
#   --resize_factor=16 \
#   --min_resize_value="${CROP_SIZE}" \
#   --max_resize_value="${MAX_RESIZE}" \
#   --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
#   --train_logdir="${TRAIN_LOGDIR}" \
#   --dataset_dir="${ADE_DATASET}" \
#   --dataset="${DATASET}"

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="${EVAL_SPLIT}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size="${MAX_RESIZE},${MAX_RESIZE}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${ADE_DATASET}" \
  --dataset="${DATASET}" \
  --max_number_of_evaluations=1

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=${ADE_CLASSES} \
  --crop_size="${CROP_SIZE}" \
  --crop_size="${CROP_SIZE}" \
  --inference_scales=1.0

