# -----------------------------------------------------------------------------
# Automatic configuration for training on the Guitar dataset
# -----------------------------------------------------------------------------
# Read the istruction carefully and modify what's needed. Be ware of the path,
# considering that this script could be generally be launched from its parent
# directory (~/workspace/semseg_mit).
# Reference values of ade20k-hrnetv2.yaml (to be modified on training success)
# -----------------------------------------------------------------------------
from yacs.config import CfgNode as CN
from yacs.config import load_cfg
import argparse
import string

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, 
                    help='Name of the pretrained model')
parser.add_argument("--cfg_file", required=True,
                    help='Path of .yaml configuration file')
args = parser.parse_args()

# Preparing a new node
_C = CN()
# # Checkpoint path of the active model
# _C.DIR = "ckpt/guitar/%s" %(args.model_name)
# Directory where the initial pretrained model is saved and where
# (theoretically) further results are stored
_C.DIR = "ckpt/%s" %(args.model_name)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "data/guitar/"
_C.DATASET.list_train = "data/guitar/training.odgt"
_C.DATASET.list_val = "data/guitar/validation.odgt"
# /!\ ignore_labels should not be included into the count...
_C.DATASET.num_class = 2
# homogeneous train/test, size of short edge (int or tuple) 
_C.DATASET.imgSizes = (720,)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 960
#  max down sampling rate of network to avoid rounding during conv or pooling
_C.DATASET.padding_constant = 32
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 4
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "hrnetv2"
# architecture of net_decoder
_C.MODEL.arch_decoder = "c1"
# weights at "" give a non-null argument error, so we don't specify
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 720

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# batch size
_C.TRAIN.batch_size_per_gpu = 4
# epochs to train for
# _C.TRAIN.num_epoch = 30
_C.TRAIN.num_epoch = 8 # num_iter_desired / num epoch_iters (20K = 4; 40K = 8; 80K = 16)
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 1 # just to set verbose
# manual seed
_C.TRAIN.seed = 304

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# output visualization during validation (like in teaser)
_C.VAL.visualize = True
# the checkpoint to evaluate on (put the number of last epoch)
_C.VAL.checkpoint = "epoch_8.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on (in defaults, it is not clear where it is put)
_C.TEST.checkpoint = "epoch_8.pth"
# folder to output visualization results
_C.TEST.result = "./result/guitar/exp04"

# -----------------------------------------------------------------------------
# Apply defaults
# -----------------------------------------------------------------------------
def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

if __name__ == "__main__":
  cfg = get_cfg_defaults()
  cfg.merge_from_file(args.cfg_file)
  cfg.freeze()
  print('Saving configuration to: ', args.cfg_file)
  with open(args.cfg_file, 'w') as cfg_file:
    cfg_file.write(str(cfg))

  # print(cfg)