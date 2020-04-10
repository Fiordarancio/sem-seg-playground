#!/bin/bash
#---------------------------------------------------------------------------
# Latest update: march 2020
#---------------------------------------------------------------------------
# Using:
#   hrnet+c1
# as models to evaluate on the guitar dataset. The model is taken from the
# previous training over an hrnet-c1 accoring to relative configuration.

# Usage:
#   $ ./guitar_eval_hrnet.sh [|& tee logs/gevalx.txt]
# where x is the id number of the experiment.
#---------------------------------------------------------------------------

set -e

echo "Guitar evaluation HRNET - $(date)"
echo "--------------"
echo "Configuring..."
# Model names
DATASET=guitar
PRETRAINED_MODEL=ade20k-hrnetv2-c1
MODEL=$DATASET-hrnetv2-c1
MODEL_PATH=ckpt/$MODEL

echo "Dataset: ${DATASET}"
echo "Model path: ${MODEL_PATH}"

# Config file
CONFIG_FILE=./config/$MODEL.yaml
echo "Config file: ${CONFIG_FILE}"

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# export CUDA_VISIBLE_DEVICES 
export CUDA_VISIBLE_DEVICES="0,3"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"
# Gpus to pass must be indexes of actually visible devices
GPUS=0,1
# consider using:
set CUDA_LAUNCH_BLOCKING=1

echo "----------------------"
echo "Starting evaluation..."
echo "$(date)"
# Training
python3 eval_multipro.py \
  --cfg $CONFIG_FILE \
  --gpus $GPUS \
  # DIR $MODEL_PATH \

echo "--------------------------------"
echo "$(date)"
