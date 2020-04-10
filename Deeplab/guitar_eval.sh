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

# This script is used to evaluate the results of the training done on the
# Guitar dataset. Note that flags passed to eval.py MUST MATCH the ones 
# used in the training whose analysis we are interested in.
#
# Usage:
#		# From the tensorflow/models/research/deeplab directory.
#		sh ./guitar_eval.sh [|& tee logs/guitar/<date_of_launch>/guitar_eval.txt]
#	
# Please note that when passing flags to the following Python scripts, relative
# strings should change into the file, so be careful in avoiding overwritings.

echo "DEEPLABv3+ evaluation experiment launched on $(date)"

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Set up the working environment.
CURRENT_DIR=$(pwd) # models/research
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set the dataset folder: data are ready, no need to download anything
DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# export CUDA_VISIBLE_DEVICES in order to select which of our GPUs 
# must be used. Consider reverse order than nvidia-smi.
export CUDA_VISIBLE_DEVICES="3"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"

# Set up the splits: dataset on which we are working.
TRAIN_SPLIT="train_aug"
EVAL_SPLIT="eval_aug"
MODEL_VARIANT="xception_65"
# Set the num_iterations correspondent to the model to evaluate.
NUM_ITERATIONS=40000
# Crop sizes to be used in evaluation. You need whole-image inference, so
# brute force value is [max_width, max_height] + 1; suggested value is 
# output_stride * 2^k + 1. See DEBUG NOTES below for further details.
CROP_W=961
CROP_H=721

# Set up the working directories.
GUITAR_FOLDER="guitars"
# EXP_FOLDER="exp/train_on_spade_aug_set"
# EXP_FOLDER="exp/working_on_${TRAIN_SPLIT}_${EVAL_SPLIT}_${NUM_ITERATIONS}"
EXP_FOLDER="exp/working_on_${TRAIN_SPLIT}_${EVAL_SPLIT}_${NUM_ITERATIONS}"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/eval"
mkdir -p "${EVAL_LOGDIR}"

GUTIAR_TFRECORD="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/tfrecord"

# # Change here if you evaluate on different split (example: trainval)
# EVAL_SPLIT="eval_aug"

echo "------------------------------"
echo "Evaluating on GUITAR folders: "
echo "    ${GUITAR_FOLDER}/${EXP_FOLDER}"
echo "TRAIN_LOGDIR: ${TRAIN_LOGDIR}"
echo "EVAL_LOGDIR: ${EVAL_LOGDIR}"
echo "Train split for checkpoint: ${TRAIN_SPLIT}"
echo "Eval split for evaluation: ${EVAL_SPLIT}"
echo "Num iterations: ${NUM_ITERATIONS}"
echo "Crop sizes (w,h): [${CROP_H}, ${CROP_W}]"
echo "Atrous rates: [6, 12, 18]"
echo "Output stride: 16"
echo "Decoder output stride: 4"
echo "Max number of evaluations: 1 (more could stuck unpredictably)"
echo "    Non printed values as kept as default. See eval.py"

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
# DEBUG NOTES (if you encounter issues): 
# 1) we have to do whole-image inference, meaning you need to set eval_crop_size
#    greater or equal than the largest image dimension. Usually, authors use a 
#    crop_size = output_stride * k + 1, where k is an integer (usually, 
#    dimensions follow powers of 2). For this reason we generally put 1024,1024,
#    which is a range that appears to fit our needs (changes depending on dataset)
# 3) as a consequence, it's again VERY important to have a coherent dataset, if 
#    possible. This means having images and labels of equal size and aspect ratio.
# 4) notice that the aforementioned size should have a 2^n form
# 5) useful referenced Github issues #107 #3730 #4203 #3906 of Deeplab 
echo "----------------------------------------------------------------"
echo "Starting evaluation on $(date)"
python "${WORK_DIR}"/eval.py \
 --logtostderr \
 --eval_split="${EVAL_SPLIT}" \
 --model_variant="${MODEL_VARIANT}" \
 --atrous_rates=6 \
 --atrous_rates=12 \
 --atrous_rates=18 \
 --output_stride=16 \
 --decoder_output_stride=4 \
 --eval_crop_size="${CROP_H},${CROP_W}" \
 --checkpoint_dir="${TRAIN_LOGDIR}" \
 --eval_logdir="${EVAL_LOGDIR}" \
 --dataset_dir="${GUTIAR_TFRECORD}" \
 --max_number_of_evaluations=1 \
 --dataset="${GUITAR_FOLDER}" \

echo "---------------------------------------------------------"
echo "Evaluation completed on $(date)"
echo "Check miou using:"
echo "    $ tensorboard --logdir ${EVAL_LOGDIR} --host localhost"
