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

# This script is used to run local test on Guitars dataset.
#
# Usage:
#		# From the tensorflow/models/research/deeplab directory.
#		sh ./guitar_train.sh [|& tee logs/guitar_train.sh]
#	
# If you previously interrupted the training, we suggest to refer to:
#   guitar_recover_train.sh
#
# Please note that when passing flags to the following Python scripts, relative
# strings should change into the file, so be careful in avoiding overwritings.
#-------------------------------------------------------------------------------

echo "DEEPLABv3+ training experiment launched on $(date)"

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Set up the working environment.
CURRENT_DIR=$(pwd) # models/research
WORK_DIR="${CURRENT_DIR}/deeplab"

## Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py # -v # this flag arises errors...
#echo "PYTHONPATH correctly set."

# Set the dataset folder: data are ready, no need to download anything
DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# export CUDA_VISIBLE_DEVICES in order to select which of our GPUs 
# must be used. It appears to read the indexed opposite than nvidia-smi
export CUDA_VISIBLE_DEVICES="0,3"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"

# Train iterations: steps of training. Epochs can be extracted this way
# steps_needed_to_traverse_dataset = dataset_size / batch_size
# epochs = num_iterations / steps_needed_to_traverse_dataset

# 2019-12-03/04 = 2019-12-07
# NUM_ITERATIONS=40000 # scaling up to 80K
NUM_ITERATIONS=40000
BATCH_SIZE=8 # maximum for 1 GPU only is 8
FINETUNE_BN=False
CROP_SIZE=513
NUM_CLASSES=2
#---------------------------------------------------------------------------
# Legacy.
# # 2019-11-04/05
# NUM_ITERATIONS=20000
# TRAIN_SPLIT="spade_aug_train"
# CROP_SIZE=256 # 513
# # 2019-10-25
# NUM_ITERATIONS=30000
# CROP_SIZE=513
# # 2019-10-19 
# NUM_ITERATIONS=20000
# # 2019-10-18/19 
# NUM_ITERATIONS=10000
# --train-split="trainval_aug"
# # 2019-10-16/18 
# NUM_ITERATIONS=20000 
# --train-split="trainval_aug"
#---------------------------------------------------------------------------

# Set the splits: dataset on which we are working.
# # 2019-12-07
# TRAIN_SPLIT="trainval"
# EVAL_SPLIT="trainval"
# 2019-12-03
TRAIN_SPLIT="train_aug"
EVAL_SPLIT="eval_aug"
MODEL_VAR="xception_65" # network backbone model variant
# PRETRAINED="PASCAL-COCO"
PRETRAINED="${DATASET} ${TRAIN_SPLIT} ${NUM_ITERATIONS}"

# Set up the working directories.
GUITAR_FOLDER="guitars"
# EXP_FOLDER="exp/train_on_trainval_aug_set"
# EXP_FOLDER="exp/train_on_train_val_aug_set"
EXP_FOLDER="exp/working_on_${TRAIN_SPLIT}_${EVAL_SPLIT}_${NUM_ITERATIONS}"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/train"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

#---------------------------------------------------------------------
# Choose the initial checkpoint:
# 1) if you start the finetuning from zero, use a model from the list 
# of provided ones in g3doc/model_zoo.
# 2) if you fine tune your own model with more iterations, comment
#    this first option and put the path to the last checkpoint as
#    your target.
#---------------------------------------------------------------------
# 1) Copy locally the trained checkpoint as the initial checkpoint.
# All our experiments start, at first, with the PASCAL VOC pretrained
# network (over xception_65)
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/init_models"
mkdir -p "${INIT_FOLDER}"
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
TF_INIT_CKPT="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt"

# # 2) Indicate the path of your target --> use as recover_train.sh
# INIT_FOLDER=${TRAIN_LOGDIR}
# TF_INIT_CKPT="${INIT_FOLDER}/checkpoint"

echo "WARNING: If you want to finetune a previously trained model up to more iterations,"
echo "         remember to import the corresponding checkpoints into this exp folder."

cd "${CURRENT_DIR}"

# Path to the input dataset in format tfrecord (be sure that 
# build_[name]_dataset.py has been launched and info in 
# data_generator.py are correct)
GUTIAR_TFRECORD="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/tfrecord"

echo "------------------------"
echo "Created GUITAR folders: "
echo "    ${GUITAR_FOLDER}/${EXP_FOLDER}"
echo "Initial checkpoints:    ${INIT_FOLDER}"
echo "Log for training:       ${TRAIN_LOGDIR}"
echo "Export results in:      ${EXPORT_DIR}"

#----------------------------------------------------------------------
# Summary and training.
echo "----------------------------"
echo "Starting training. Options:"
echo "Model variant: ${MODEL_VAR}"
echo "Finetuning model from ${PRETRAINED}: ${TF_INIT_CKPT}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Dataset (tfrecord): ${GUTIAR_TFRECORD}"
echo "Dataset split: ${TRAIN_SPLIT}"
echo "Num classes of dataset: ${NUM_CLASSES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Finetuning the Batch Norm layers: ${FINETUNE_BN}"
echo "Atrous rates: [6, 12, 18]"
echo "Output stride: 16"
echo "Deconder output stride: 4"
echo "Crop size: ${CROP_SIZE}"
echo "    Non printed values as kept as default. See train.py"
echo "    For more pretrained nets and variants, see ./g3doc/model_zoo.md"

echo "Start time: $(date)"

python "${WORK_DIR}"/train.py \
 --logtostderr \
 --train_split="${TRAIN_SPLIT}" \
 --model_variant="${MODEL_VAR}" \
 --atrous_rates=6 \
 --atrous_rates=12 \
 --atrous_rates=18 \
 --output_stride=16 \
 --decoder_output_stride=4 \
 --train_crop_size="${CROP_SIZE},${CROP_SIZE}" \
 --train_batch_size=${BATCH_SIZE} \
 --training_number_of_steps="${NUM_ITERATIONS}" \
 --tf_initial_checkpoint="${TF_INIT_CKPT}" \
 --train_logdir="${TRAIN_LOGDIR}" \
 --dataset_dir="${GUTIAR_TFRECORD}" \
 --dataset="${GUITAR_FOLDER}" \
 --fine_tune_batch_norm=${FINETUNE_BN} \
 --initialize_last_layer=False \
 --last_layers_contain_logits_only=False
#  --min_resize_value=${CROP_SIZE} \
#  --max_resize_value=${CROP_SIZE} \
# --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \

echo "End time: $(date)"

#----------------------------------------------------------------------
# Export the trained checkpoint.
echo "--------------------------------"
echo "Export the trained checkpoint..."

CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="${MODEL_VAR}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=$NUM_CLASSES \
  --crop_size=$CROP_SIZE \
  --crop_size=$CROP_SIZE \
  --inference_scales=1.0

# Move the final checkpoints in the export directory.
# cp "${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}.*" -t ${EXPORT_DIR}

echo "-----------------------------------------------------"
echo "Deeplab completed. Check results running tensorboard:" 
echo "    $ tensorboard --logdir ${TRAIN_LOGDIR} --host localhost"
