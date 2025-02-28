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

""" Create TFRecord-based dataset for EventLab MoTIVE dataset

Following the ADE20K script, since we have the examples at hand
"""

import math
import os
import random
import sys
import build_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

'''
Validation images should be split from the entire set programmatically (?)
'''

DATASETS = '/home/ilaria/workspace/models/research/deeplab/datasets/'
MOTIVE = os.path.join(DATASETS, 'MoTIVE')
MOTIVE_IMG = os.path.join(MOTIVE, 'images_aug')
MOTIVE_LAB = os.path.join(MOTIVE, 'labels_aug')
MARK_KNOPFLER = os.path.join(DATASETS, 'guitars')

# define folders where finding training and evaluation data
tf.app.flags.DEFINE_string(
    'train_images',
    os.path.join(MOTIVE_IMG, 'train'),
    'Folder containing training images')
    
tf.app.flags.DEFINE_string(
    'train_labels',
    os.path.join(MOTIVE_LAB, 'train'),
    'Folder containing annotations for training images')

tf.app.flags.DEFINE_string(
    'eval_images',
    os.path.join(MOTIVE_IMG, 'eval'),
    'Folder containing evaluation images')
    
tf.app.flags.DEFINE_string(
    'eval_labels',
    os.path.join(MOTIVE_LAB, 'eval'),
    'Folder containing annotations for evaluation images')

# motive augmented
tf.app.flags.DEFINE_string(
    'train_images_aug',
    os.path.join(MOTIVE_IMG, 'train'),
    'Folder containing training images')
    
tf.app.flags.DEFINE_string(
    'train_labels_aug',
    os.path.join(MOTIVE_LAB, 'train'),
    'Folder containing annotations for training images')

tf.app.flags.DEFINE_string(
    'eval_images_aug',
    os.path.join(MOTIVE_IMG, 'eval'),
    'Folder containing evaluation images')
    
tf.app.flags.DEFINE_string(
    'eval_labels_aug',
    os.path.join(MOTIVE_LAB, 'eval'),
    'Folder containing annotations for evaluation images')

# visualization split (test split) over Mark Knopfler video
tf.app.flags.DEFINE_string(
    'vis_images',
    os.path.join(MARK_KNOPFLER, 'vis_images'),
    'Folder containing test images (video frames)')

tf.app.flags.DEFINE_string(
    'vis_labels',
    os.path.join(MARK_KNOPFLER, 'vis_labels'),
    'Folder containing fake labels for test images')

# tfrecord folder
tf.app.flags.DEFINE_string(
    'output_dir',
    os.path.join(MOTIVE, 'tfrecord'),
    'Path to save converted tfrecord of Tensorflow example')

_NUM_SHARDS = 8 #4


def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir):
  """Converts the guitar dataset into into tfrecord format.

  Args:
    dataset_split: Dataset split (e.g., train, val, trainval (?)).
    dataset_dir: Dir in which the dataset locates.
    dataset_label_dir: Dir in which the annotations locates.

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """

  img_names = tf.gfile.Glob(os.path.join(dataset_dir, '*.jpg'))
  random.shuffle(img_names)
  seg_names = []
  for f in img_names:
    # get the filename without the extension
    basename = os.path.basename(f).split('.')[0]
    # cover its corresponding *_seg.png
    seg = os.path.join(dataset_label_dir, basename+'.png')
    seg_names.append(seg)

  print('Num read labels ', len(seg_names))
  num_images = len(img_names)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = img_names[i]
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = seg_names[i]
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, img_names[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  tf.gfile.MakeDirs(FLAGS.output_dir)
  # # train split
  # _convert_dataset('train', FLAGS.train_images, FLAGS.train_labels)
  # eval split
  # _convert_dataset('eval', FLAGS.eval_images, FLAGS.eval_labels)
  # external vis split
  _convert_dataset('vis', FLAGS.vis_images, FLAGS.vis_labels)
  # train_aug / eval_aug split
  _convert_dataset('train_aug', FLAGS.train_images_aug, FLAGS.train_labels_aug)
  _convert_dataset('eval_aug', FLAGS.eval_images_aug, FLAGS.eval_labels_aug)

if __name__ == '__main__':
  tf.app.run()
