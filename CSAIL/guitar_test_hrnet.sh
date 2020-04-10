#!/bin/bash
#---------------------------------------------------------------------------
# Latest update: march 2020
#---------------------------------------------------------------------------
# Using:
#   hrnet 
# as models to test the model previously trained on the guitar dataset. 
# Configuration data is put in the same file used for training.
#
# Usage:
#   $ ./guitar_test_hrnet.sh [|& tee logs/gtest<#test>.txt]
#---------------------------------------------------------------------------

set -e

echo "Guitar test HRNET - $(date)"
echo "--------------"
echo "Configuring..."
# Model names
DATASET=guitar
MODEL=$DATASET-hrnetv2-c1
MODEL_PATH=ckpt/$MODEL

echo "Dataset: ${DATASET}"
echo "Model path: ${MODEL_PATH}"
# Path to test images
TEST_IMGS_PATH=./data/$DATASET/images/testing
echo "Generating images from: ${TEST_IMGS_PATH}"

# Config file
CONFIG_FILE=./config/$MODEL.yaml
echo "Config file: ${CONFIG_FILE}"

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# export CUDA_VISIBLE_DEVICES in order to select which of our GPUs 
# must be used. It can also be used to mask GPUs. For testing, only
# one GPU is required (error otherwise)
export CUDA_VISIBLE_DEVICES="0,3"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"
# Gpus to pass must be indexes of actually visible devices
GPUS=1

echo "--------------------"
echo "Starting testing..."
echo "$(date)"
# Training
python3 -u test.py \
  --img $TEST_IMGS_PATH \
  --cfg $CONFIG_FILE \
  --gpu $GPUS \
  # DIR $MODEL_PATH \

echo "---------------------------------"
echo "$(date)"
echo "Test completed. Check results"
