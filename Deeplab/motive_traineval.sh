#!/bin/bash
# 2019 Event Lab
#-------------------------------------------------------------------------------
# Run training and evaluation of Deeplab of the MoTIVE minidataset.
#
# Usage:
#		# From the tensorflow/models/research/deeplab directory.
#		sh motive_traineval.sh [ |& tee logs/motive_traineval.txt ]
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

# Check PYTHONPATH and installed bash version
echo "Check PYTHONPATH: ${PYTHONPATH}"
echo "Check BASH_VERSION: ${BASH_VERSION}"

# Set the dataset folder: data are ready, no need to download anything
DATASET_DIR="datasets"
# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories and datasets.
MOTIVE_FOLDER="MoTIVE"
MOTIVE_DATASET="motive"
MOTIVE_TFRECORD="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/tfrecord"
MOTIVE_CLASSES=35

EXP_FOLDER="exp/train_eval"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

echo "This dadaset's folder:  ${MOTIVE_FOLDER}/${EXP_FOLDER}"
echo "Initial checkpoints:    ${INIT_FOLDER}"
echo "Log for training:       ${TRAIN_LOGDIR}"
echo "Log for evaluation:     ${EVAL_LOGDIR}"
echo "Log for visualization:  ${VIS_LOGDIR}"
echo "Export results in:      ${EXPORT_DIR}"

TRAIN_SPLIT="train"
EVAL_SPLIT="eval"

TRAIN_SPLIT_SIZE=42
EVAL_SPLIT_SIZE=10
echo "Train split name: ${TRAIN_SPLIT}; size: ${TRAIN_SPLIT_SIZE}"
echo "Eval split name: ${EVAL_SPLIT}; size: ${EVAL_SPLIT_SIZE}"

# Copy locally the trained checkpoint as the initial checkpoint. Still,
# we are using the pretrained Pascal checkpoint. For other datasets, 
# please check g3doc/modelzoo.md
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

# Select the training options.
# Train iterations: steps of training. Epochs can be extracted this way
# steps_needed_to_traverse_dataset = dataset_size / batch_size
# epochs = num_iterations / steps_needed_to_traverse_dataset
NUM_ITERATIONS=60000
BATCH_SIZE=4
# The crop size should be used as resize value only if the size of 
# your input images are fixed and known, as well as their aspect ratio.
CROP_SIZE=513
TRAIN_CROP_SIZE="${CROP_SIZE}, ${CROP_SIZE}"

# Write down the applied metrics to check mispellings.
echo "-------"
echo "Options"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Training crop size: ${TRAIN_CROP_SIZE}"

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# Export CUDA_VISIBLE_DEVICES in order to select which of our GPUs must be used.
# Check the correctness of the index using $ nvidia-smi
export CUDA_VISIBLE_DEVICES="0" 
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"
echo "(device = 0 : using GeForce; device = 1 : using Quadro)"

# Train.
# echo "--------------------"
# echo "Starting training..."
# echo $(date)

# python "${WORK_DIR}"/train.py \
#   --logtostderr \
#   --train_split="${TRAIN_SPLIT}" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --train_crop_size="${TRAIN_CROP_SIZE}" \
#   --train_batch_size=${BATCH_SIZE} \
#   --training_number_of_steps="${NUM_ITERATIONS}" \
#   --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
#   --train_logdir="${TRAIN_LOGDIR}" \
#   --dataset_dir="${MOTIVE_TFRECORD}" \
#   --dataset=${MOTIVE_DATASET} \
#   --initialize_last_layer=False \
#   --last_layers_contain_logits_only=False \
#   --fine_tune_batch_norm=False \

# # num_clones=1 # using only the Quadro GPU
# echo "Training finished on" $(date)

# # Evaluation. Consider that some images are HD so maybe we require
# # a bigger eval_crop_size.
# echo "----------------------"
# echo "Starting evaluation..."
# echo $(date)

# # Eval crop size should be always be higher than the highest sizes,
# # which, in the case of the MoTIVE minidataset, are as maxima
# # width: 3000 and height: 2000. The authors suggest to put just a 
# # pixel more of the maximum dimensions.
# EVAL_CROP_SIZE="2001, 3001" 

# python "${WORK_DIR}"/eval.py \
# --logtostderr \
# --eval_split="${EVAL_SPLIT}" \
# --model_variant="xception_65" \
# --atrous_rates=6 \
# --atrous_rates=12 \
# --atrous_rates=18 \
# --output_stride=16 \
# --decoder_output_stride=4 \
# --eval_crop_size="${EVAL_CROP_SIZE}" \
# --checkpoint_dir="${TRAIN_LOGDIR}" \
# --eval_logdir="${EVAL_LOGDIR}" \
# --dataset_dir="${MOTIVE_TFRECORD}" \
# --max_number_of_evaluations=1 \
# --dataset=${MOTIVE_DATASET} \
# #  --eval_crop_size="513,513" \ 

# echo "Evaluation finished on" $(date)

# Export the trained checkpoint
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

echo "-------------------------------"
echo "Exporting trained checkpoint..."

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
  --num_classes=$MOTIVE_CLASSES \
  --crop_size=$CROP_SIZE \
  --crop_size=${CROP_SIZE} \
  --inference_scales=1.0 \

# Run Tensorboard or for evaluation. We suggest to use, outside
# the scope of this code:
#   $ tensorboard --logdir [TRAIN_LOGDIR] --host localhost
# for training; while for evaluation:
#   $ tensorboard --logdir [EVAL_LOGDIR] --host localhost
