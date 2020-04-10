#!/bin/bash
#---------------------------------------------------------------------------
# Latest update: march 2020
#---------------------------------------------------------------------------
# Using:
#   hrnet - c1
# as models to train on the guitar dataset. Default values used for
# configuration can be found into a python file into ./config/ which has the 
# same name of the model. If you launch this script, a .yaml file will be 
# created with the needed configuration. Each option can be overwritten also
# by command line.
#
# Usage:
#   $ ./guitar_train.sh [|& tee logs/guitar_train.txt]
#---------------------------------------------------------------------------

set -e

echo "Guitar training HRNET - $(date)"
echo "--------------"
echo "Configuring..."
# Model names
DATASET=guitar
PRETRAINED_MODEL=ade20k-hrnetv2-c1
MODEL=$DATASET-hrnetv2-c1
# MODEL=$PRETRAINED_MODEL
MODEL_PATH=ckpt/$MODEL

echo "Dataset: ${DATASET}"
echo "Pretrained model: ${PRETRAINED_MODEL}"
echo "Model path: ${MODEL_PATH}"

# Config file
CONFIG_FILE=./config/$MODEL.yaml
echo "Config file: ${CONFIG_FILE}"
# Remove and recreate file if present
rm -f ${CONFIG_FILE}
touch $CONFIG_FILE
python ./config/defaults_${DATASET}_hrnet.py --model_name $MODEL --cfg_file $CONFIG_FILE

PATHENC=$MODEL_PATH/encoder_epoch_30.pth
PATHDEC=$MODEL_PATH/decoder_epoch_30.pth
ENCODER=$PRETRAINED_MODEL/encoder_epoch_30.pth
DECODER=$PRETRAINED_MODEL/decoder_epoch_30.pth

# Download model weights when starting from original models
if [ ! -e $MODEL_PATH ]; then
  mkdir $MODEL_PATH
  echo "Created: ${MODEL_PATH}"
fi
if [ ! -e $PATHENC ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $PATHDEC ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# export CUDA_VISIBLE_DEVICES in order to select which of our GPUs 
# must be used. It can also be used to mask GPUs. For training, they
# suggest to use >= 4 GPUs. For us, working on Quadro+GeForce works
# but it is quite slow
export CUDA_VISIBLE_DEVICES="0,3"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"
# Gpus to pass must be indexes of actually visible devices
GPUS=0,1

echo "----------------------------------------------------"
echo "Training started on $(date)"
# Training
python train.py \
  --cfg $CONFIG_FILE \
  --gpus $GPUS \
  # DIR $MODEL_PATH \

echo "----------------------------------------------------"
echo "Training completed on $(date)"
