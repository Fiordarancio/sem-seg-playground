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

echo "DEEPLABv3+ experiment launched on $(date)"

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..
# Set up the working environment.
CURRENT_DIR=$(pwd) # models/research
WORK_DIR="${CURRENT_DIR}/deeplab"

# Check PYTHONPATH and installed bash version
echo "Check PYTHONPATH: ${PYTHONPATH}"

# Set the dataset folder: data are ready, no need to download anything
DATASET_DIR="datasets"
# Go back to original directory.
cd "${CURRENT_DIR}"

# Check CUDA and cuDNN library path are correctly loaded.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
echo "Loaded CUDA lybrary: ${LD_LIBRARY_PATH}"
# Export CUDA_VISIBLE_DEVICES in order to select which of our GPUs must be used.
# Check the correctness of the index using $ nvidia-smi
export CUDA_VISIBLE_DEVICES="1" 
echo "Cuda visible devices = ${CUDA_VISIBLE_DEVICES}"
echo "(device = 0 : using GeForce; device = 1 : using Quadro)"

# Select the training options.
# Train iterations: steps of training. Epochs can be extracted this way
# steps_needed_to_traverse_dataset = dataset_size / batch_size
# epochs = num_iterations / steps_needed_to_traverse_dataset
NUM_ITERATIONS=80000 # scaling up to 80K
BATCH_SIZE=8
FINETUNE_BN=False
# The crop size should be used as resize value only if the size of 
# your input images are fixed and known, as well as their aspect ratio.
CROP_SIZE=513
TRAIN_CROP_SIZE="${CROP_SIZE}, ${CROP_SIZE}"
NUM_CLASSES=35

# Set the splits: dataset on which we are working.
TRAIN_SPLIT="train"
EVAL_SPLIT="eval"
VIS_SPLIT="vis"
MODEL_VAR="xception_65" # network backbone model variant
# PRETRAINED="PASCAL-COCO"
PRETRAINED="motive/trainval 40K"

# Set up the working directories and datasets.
MOTIVE_FOLDER="MoTIVE"
MOTIVE_DATASET="motive"
MOTIVE_TFRECORD="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/tfrecord"

EXP_FOLDER="exp/working_on_${TRAIN_SPLIT}_${EVAL_SPLIT}_${NUM_ITERATIONS}"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/eval"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/export"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/${EXP_FOLDER}/vis"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${EXPORT_DIR}"
mkdir -p "${VIS_LOGDIR}"

# #---------------------------------------------------------------------
# # Choose the initial checkpoint:
# # 1) if you start the finetuning from zero, use a model from the list 
# # of provided ones in g3doc/model_zoo.
# # 2) if you fine tune your own model with more iterations, comment
# #    this first option and put the path to the last checkpoint as
# #    your target.
# #---------------------------------------------------------------------
# # 1) Copy locally the trained checkpoint as the initial checkpoint.
# # All our experiments start, at first, with the PASCAL VOC pretrained
# # network (over xception_65)
# INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${MOTIVE_FOLDER}/init_models"
# mkdir -p "${INIT_FOLDER}"
# TF_INIT_ROOT="http://download.tensorflow.org/models"
# TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
# cd "${INIT_FOLDER}"
# wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
# tar -xf "${TF_INIT_CKPT}"
# TF_INIT_CKPT="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt"

# 2) Indicate the path of your target --> use as recover_train.sh
INIT_FOLDER=${TRAIN_LOGDIR}
TF_INIT_CKPT="${INIT_FOLDER}/checkpoint"

echo "WARNING: If you want to finetune a previously trained model up to more iterations,"
echo "         remember to import the corresponding checkpoints into this exp folder."

cd "${CURRENT_DIR}"

echo "------------------------"
echo "Created MOTIVE folders: "
echo "    ${MOTIVE_FOLDER}/${EXP_FOLDER}"
echo "Initial checkpoints:    ${INIT_FOLDER}"
echo "Log for training:       ${TRAIN_LOGDIR}"
echo "Log for evaluation:     ${EVAL_LOGDIR}"
echo "Export results in:      ${EXPORT_DIR}"
echo "Visualize inference in: ${VIS_LOGDIR}"

#----------------------------------------------------------------------
# Summary and training.
echo "----------------------------"
echo "Starting training. Options:"
echo "Model variant: ${MODEL_VAR}"
echo "Finetuning model from ${PRETRAINED}: ${TF_INIT_CKPT}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Dataset (tfrecord): ${MOTIVE_TFRECORD}"
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
 --dataset_dir="${MOTIVE_TFRECORD}" \
 --dataset="${MOTIVE_DATASET}" \
 --fine_tune_batch_norm=${FINETUNE_BN} \
 --initialize_last_layer=False \
 --last_layers_contain_logits_only=False

echo "-----------------------------------------------------------"
echo "Starting evaluation on $(date)"
# Evaluation images should have fixed size. Put the crop_size as
# max_size +1 for each dimension. Remeber the sizes must be in 
# matrix order!! (height, width)
EVAL_CROP_SIZE="1081, 1441" 

python "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="${EVAL_SPLIT}" \
--model_variant="${MODEL_VAR}" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size="${EVAL_CROP_SIZE}" \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${MOTIVE_TFRECORD}" \
--max_number_of_evaluations=1 \
--dataset=${MOTIVE_DATASET} \

echo "----------------------------------------------------------"
echo "$(date)"
echo "Evaluation completed. Check miou using:"
echo "    $ tensorboard --logdir ${EVAL_LOGDIR} --host localhost"

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

echo "-------------------------------"
echo "Exporting trained checkpoint..."

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
  --inference_scales=1.0 \

# Run inference. Mind the crop size 
# (follows the same rules as evaluation)
echo "------------------------------------------"
echo "Running inference on split ${VIS_SPLIT}..."

EVAL_CROP_SIZE="721, 961" # if using vis

python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="${VIS_SPLIT}" \
  --model_variant="${MODEL_VAR}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --resize_factor=16 \
  --decoder_output_stride=4 \
  --vis_crop_size="${EVAL_CROP_SIZE}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${MOTIVE_TFRECORD}" \
  --dataset="${MOTIVE_DATASET}" \
  --max_number_of_iterations=1 \
  --also_save_raw_predictions=True \

echo "-----------------------------------------------------"
echo "Deeplab completed. Check results running tensorboard!"
