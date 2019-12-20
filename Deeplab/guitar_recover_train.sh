#!/bin/bash
#-------------------------------------------------------------------------------
# Update and launch this script to rapidly (and safely) recover from a
# previously interrupted training.
#-------------------------------------------------------------------------------

echo "DEEPLABv3+ experiment RECOVERED on $(date)"

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
# must be used. It appears that Deeplab does the indexing by calling
# 0 the internal GPU, while 1 and 2 Pascal and GeForce respectively.
export CUDA_VISIBLE_DEVICES="1"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"
echo "(device = 0 : using GeForce; device = 1 : using Quadro)"

# 2019-12-04/07
NUM_ITERATIONS=40000 # scaling up to 80K
BATCH_SIZE=8
FINETUNE_BN=False
CROP_SIZE=513
NUM_CLASSES=2

# Set the splits: dataset on which we are working.
# 2019-12-07
TRAIN_SPLIT="trainval"
EVAL_SPLIT="trainval"
# # 2019-12-04
# TRAIN_SPLIT="train"
# EVAL_SPLIT="eval_aug"
MODEL_VAR="xception_65" # network backbone model variant
PRETRAINED="PASCAL-COCO"

# Set up the working directories.
GUITAR_FOLDER="guitars"
# EXP_FOLDER="exp/train_on_trainval_aug_set"
# EXP_FOLDER="exp/train_on_train_val_aug_set"
EXP_FOLDER="exp/working_on_${TRAIN_SPLIT}_${EVAL_SPLIT}_${NUM_ITERATIONS}"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/export"

GUITAR_DATASET="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/tfrecord"

#----------------------------------------------------------------------
# Recover from last checkpoint. Remember that the number of iterations
# to be exectuted must be higher than the last reached by the 
# checkpoint (briefly, the max number of iteration is always good)
TF_INIT_CKPT="${TRAIN_LOGDIR}/checkpoint"
#----------------------------------------------------------------------


#----------------------------------------------------------------------
# Summary and training.
echo "--------------------"
echo "Working on folders: "
echo "   ${GUITAR_FOLDER}/${EXP_FOLDER}"
echo "TRAIN_LOGDIR: ${TRAIN_LOGDIR}"
echo "EVAL_LOGDIR: ${EVAL_LOGDIR}"
echo "VIS_LOGDIR: ${VIS_LOGDIR}"
echo "EXPORT_DIR: ${EXPORT_DIR}"

echo "--------------------------------------"
echo "Starting recorvered training. Options:"
echo "Model variant: ${MODEL_VAR}"
echo "Recovering model from: ${TF_INIT_CKPT}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Dataset: ${GUITAR_DATASET}"
echo "Dataset split: ${TRAIN_SPLIT}"
echo "Num classes of dataset: ${NUM_CLASSES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Finetuning the Batch Norm layers: ${FINETUNE_BN}"
echo "Atrous rates: [6, 12, 18]"
echo "Output stride: 16"
echo "Deconder output stride: 4"
echo "Crop size: ${CROP_SIZE}"
echo "    Non printed values as kept as default. See train.py"

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
 --training_number_of_steps=${NUM_ITERATIONS} \
 --tf_initial_checkpoint="${TF_INIT_CKPT}" \
 --train_logdir="${TRAIN_LOGDIR}" \
 --dataset_dir="${GUITAR_DATASET}" \
 --dataset="${GUITAR_FOLDER}" \
 --fine_tune_batch_norm=${FINETUNE_BN} \
 --initialize_last_layer=False \
 --last_layers_contain_logits_only=False
#  --min_resize_value=${CROP_SIZE} \
#  --max_resize_value=${CROP_SIZE} \

#----------------------------------------------------------------------
# Export the trained checkpoint.
echo "--------------------------------"
echo "Export the trained checkpoint..."
echo "$(date)"

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

echo "-----------------------------------------------------"
echo "Deeplab completed. Check results running tensorboard!"
# Run Tensorboard  
# tensorboard --logdir="${TRAIN_LOGDIR}" --host="localhost"
