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

# This script runs visualization on a given model checkpoint, already trained.
#
# Usage:
#		# From the tensorflow/models/research/deeplab directory.
#		sh ./guitar_vis.sh
#	
# Please note that when passing flags to the following Python scripts, relative
# strings should change into the file, so be careful in avoiding overwritings.
#-------------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Set up the working environment.
CURRENT_DIR=$(pwd) # models/research
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set the dataset folder: data are ready, no need to download anything.
DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# export CUDA_VISIBLE_DEVICES in order to select which of our GPUs 
# must be used. It appears that Deeplab does the indexing by calling
# 0 the internal GPU, while 1 and 2 Pascal and GeForce respectively.
export CUDA_VISIBLE_DEVICES="1"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"
echo "(device = 0 : using GeForce; device = 1 : using Quadro)"

NUM_ITERATIONS=40000
TRAIN_SPLIT="train_aug"
EVAL_SPLIT="eval_aug"
VIS_SPLIT="vis"
# Set up the working directories.
GUITAR_FOLDER="guitars"
EXP_FOLDER="exp/working_on_${TRAIN_SPLIT}_${EVAL_SPLIT}_${NUM_ITERATIONS}"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/train"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/vis"
mkdir -p "${VIS_LOGDIR}"

echo "----------------------"
echo "Using GUITAR folders: "
echo "    ${GUITAR_FOLDER}/${EXP_FOLDER}"
echo "TRAIN_LOGDIR: ${TRAIN_LOGDIR}"
echo "VIS_LOGDIR: ${VIS_LOGDIR}"

GUITAR_DATASET="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/tfrecord"
echo "GUITAR DATASET: ${GUITAR_DATASET}"

# Visualize the results on the given split.
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="${VIS_SPLIT}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --resize_factor=16 \
  --decoder_output_stride=4 \
  --vis_crop_size="1024,1024" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${GUITAR_DATASET}" \
  --max_number_of_iterations=1 \
  --dataset="${GUITAR_FOLDER}"
