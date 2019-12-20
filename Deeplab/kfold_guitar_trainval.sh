#!/bin/bash
# 2019 Event Lab
#-------------------------------------------------------------------------------
# Run K-fold validation training on Deeplab 
#
# Usage:
#		# From the tensorflow/models/research/deeplab directory.
#		bash kfold_guitar_trainval.sh
#	Use BASH and not SH, otherwise some bash instrusction will be misunderstood.
#
# Please note that when passing flags to the following Python scripts, relative
# strings should change into the file, so be careful in avoiding overwritings.


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
echo "Check PYTHONPATH: ${PYTHONPATH}"
echo "Check BASH_VERSION: ${BASH_VERSION}"

# Set the dataset folder: data are ready, no need to download anything
DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
GUITAR_FOLDER="guitars"
KFOLDER="Kfold"
# EXP_FOLDER="exp/train_on_trainval_aug_set"
# EXP_FOLDER="exp/train_on_train_val_aug_set"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

# select the dataset
GUITAR_DATASET="guitarfold"
GUITAR_TF_DATASET="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${KFOLDER}/tfrecord"

# Train iterations: steps of training. Epochs can be extracted this way
# steps_needed_to_traverse_dataset = dataset_size / batch_size
# epochs = num_iterations / steps_needed_to_traverse_dataset
NUM_ITERATIONS=20000
BATCH_SIZE=8
CROP_SIZE=513
NUMFOLDS=10

# check CUDA and cuDNN library path are correctly loaded
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# export CUDA_VISIBLE_DEVICES in order to select which of our GPUs must be used
export CUDA_VISIBLE_DEVICES="1"
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"

# Iterate through the folds in order to perform, at each time, a training on 
# NUMFOLDS-1 data sets and evaluation on the last one. Every time we need to
# start the process from the beginning, so that we collect, in the end, a
# statistical cross-validation
for (( i=6; i<=${NUMFOLDS}; i++ ))
  do
    echo "------------------------------------------------------------------"
    FOLDNAME="fold_${i}"
    TRAIN_SPLIT="${FOLDNAME}_train"
    echo "-- TRAINING on ${TRAIN_SPLIT} --"
    EXP_FOLDER="exp/${KFOLDER}/${FOLDNAME}"
    INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/init_models"
    TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/train"
    EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/eval"
    VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/vis"
    EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${GUITAR_FOLDER}/${EXP_FOLDER}/export"
    mkdir -p "${INIT_FOLDER}"
    mkdir -p "${TRAIN_LOGDIR}"
    mkdir -p "${EVAL_LOGDIR}"
    mkdir -p "${VIS_LOGDIR}"
    mkdir -p "${EXPORT_DIR}"

    echo "This dadaset folder: "
    echo "${GUITAR_FOLDER}/${EXP_FOLDER}"
    echo "INIT_FOLDER: ${INIT_FOLDER}"
    echo "TRAIN_LOGDIR: ${TRAIN_LOGDIR}"
    echo "EVAL_LOGDIR: ${EVAL_LOGDIR}"
    echo "VIS_LOGDIR: ${VIS_LOGDIR}"
    echo "EXPORT_DIR: ${EXPORT_DIR}"

    python "${WORK_DIR}"/train.py \
    --logtostderr \
    --train_split="${TRAIN_SPLIT}" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="${CROP_SIZE},${CROP_SIZE}" \
    --train_batch_size=${BATCH_SIZE} \
    --min_resize_value=${CROP_SIZE} \
    --max_resize_value=${CROP_SIZE} \
    --training_number_of_steps="${NUM_ITERATIONS}" \
    --fine_tune_batch_norm=False \
    --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
    --train_logdir="${TRAIN_LOGDIR}" \
    --dataset_dir="${GUITAR_TF_DATASET}" \
    --dataset=${GUITAR_DATASET} \
    --initialize_last_layer=False \
    --last_layers_contain_logits_only=False \

    # num_clones=1 # using only the Quadro GPU


    # evaluate
    echo "------------------------------------------------------------------"
    EVAL_SPLIT="${FOLDNAME}_eval"
    echo "-- EVALUATING ${EVAL_SPLIT} --"

    python "${WORK_DIR}"/eval.py \
    --logtostderr \
    --eval_split="${EVAL_SPLIT}" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1024,1024" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --eval_logdir="${EVAL_LOGDIR}" \
    --dataset_dir="${GUITAR_TF_DATASET}" \
    --max_number_of_evaluations=1 \
    --dataset=${GUITAR_DATASET} \
    #  --eval_crop_size="513,513" \ 

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
      --num_classes=2 \
      --crop_size=$CROP_SIZE \
      --crop_size=$CROP_SIZE \
      --inference_scales=1.0
  done

# Run Tensorboard  
tensorboard --logdir="${TRAIN_LOGDIR}" --host="localhost"
