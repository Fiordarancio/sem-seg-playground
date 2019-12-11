#!/bin/bash
# 2019 Event Lab
#-------------------------------------------------------------------------------
# Run visualization of Deeplab of the MoTIVE minidataset. We use Mark 
# Knopfler's sequences as test set.
#
# Usage:
#		# From the tensorflow/models/research/deeplab directory.
#		sh motive_vis.sh [ |& tee logs/motive_vis.txt ]
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

VIS_SPLIT="eval"
EXP_FOLDER="exp/train_eval"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/train"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/vis_eval"

# Visualization crop size should follow the same rules of evaluation
VIS_CROP_SIZE="2001,3001" #"513,513" #"721,961" #

echo "This dadaset's folder:            ${MOTIVE_FOLDER}/${EXP_FOLDER}"
echo "Trained checkpoint used:          ${TRAIN_LOGDIR}"
echo "Predictions saved at:             ${VIS_LOGDIR}"
echo "Generating predictions on split:  ${VIS_SPLIT}"
echo "Applied crop size:                ${VIS_CROP_SIZE}"

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# Export CUDA_VISIBLE_DEVICES in order to select which of our GPUs must be used.
# Check the correctness of the index using $ nvidia-smi
export CUDA_VISIBLE_DEVICES="0" 
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"

# Visualize the results on the given split. Mind the crop size 
# (follows the same rules as evaluation)
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
  --vis_crop_size="${VIS_CROP_SIZE}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${MOTIVE_TFRECORD}" \
  --dataset="${MOTIVE_DATASET}" \
  --max_number_of_iterations=1 \
  --also_save_raw_predictions=True \
