# Step 1: install Deeplab

Clone the whole `tensorflow/models` repo from [here](https://github.com/tensorflow/models).
Create a virtual environment and use the `requirements.txt` of this directory to install dependencies fast. 
In any case, check all the steps indicated in their 
[manual](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md).

# Step 2: dataset

Download your dataset and put it into deeplab's `datasets` folder. Substitute their `data_generator.py` with our
version to get the parameters applied on our `guitar` and `motive` datasets. You can check the folders that we
uploaded here. 

Launch `build_guitar_dataset.py` or `build_motive_dataset.py` in order to convert the prepared images
into *tfrecords* which is the file format the network actually wants as input.

## Loading a brand new dataset: an example as a walkthrough
It is reccommended to divide train and evaluation data in different folders, as well as the images and the annotations. 

Remember to check that:
* the number of images and the one of the corresponding labels must be equal
* labels must have `.png` format
* labels must be 1 channel indexed annotation, i.e. every pixel in the matrix mush have a value included into the
range `[ 0, num_classes-1 ]`

Once your folders are ready, you need to convert this images in `tfrecords`, that are the data actually 
accessed by Deeplab. In order to do that, you can copy one of the `build_<datasetname>_dataset.py` and modify
it to organize your data into _splits_: these are sets of examples, subsets of the dataset; each split includes
some images and labels. This is very handy when, for example, you want to use the same total amount of data
in different percentages of training and evaluation, or you use some data with special preprocessing but 
maintaining the same kind of annotation classes.

Let's suppose we have our images and labels organized like: 
```
path-to-deeplab/datasets/mydataset
  | - train_img
  | - train_lab
  | - eval_img
  | \ eval_lab
```
and we want to define `train_split` and `eval_split` for them. Into our new `build_mydataset.py` we can use
`tf.app.flags` to easily define strings and access paths. Then we just call a proper function to convert them:
```python
# build_mydataset.py
PATH = 'path-to-deeplab/datasets/mydataset'
tf.app.flags.DEFINE_string( 
    'train_images', # name of the variable
    os.path.join(PATH, 'train_img'), # value of the variable
    'Folder containing training images of my dataset') # description
    
tf.app.flags.DEFINE_string(
    'train_labels',
    os.path.join(PATH, 'train_lab'),
    'Folder containing annotations for trainng images of my dataset')

# later, into __main__, call _convert_dataset for each split you want to create
   _convert_dataset('train_split', FLAGS.train_images, FLAGS.train_labels)
```
Once the records are built (by calling `$ python build_mydataset.py`), we need to specify the general information 
about the dataset. Into `data_generator.py` there is a list of static `DatasetDescriptor` objects. This list will 
be read by Deeplab when training, evaluating or visualizing anything. Information must be consistent. You can define 
a new element of the list with your values, like in the the following example:
```python
_MYDATASET_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train_split': 2975,    # number of elements in the image/label folders 
        'eval_split': 500,      # related to these splits
    },
    num_classes=19, # classes of your annotations. Here, indexes in labels must have value 0-18
    ignore_label=255, # value of a pixel in annotations that will be ignored while training
)
```

Take care of the `ignore_label` value. If you don't have a background/undefined class, like in the example above,
it means that all the annotation pixels will be relevant during training. So, put `ignore_label >= num_classes` to
exclude any value. This happens with our `guitars` dataset.

Besides, if you have a background class and you want to exclude pixels marked with that class from the training, 
you have to put the correspondent value. For example, there could be some 'black' pixels with value `0` marked as 
`Background`, which is indeed the first class: then it should be `ignore_label=0`. This happens with our `motive` dataset.

# Step 3: launch

Enable the virtual environment that you created for dealing with Deeplab.

Deeplab splits the pipeline in four parts: training, evaluation, visualization (inference) and export of a
trained model. Here are some bash files, whose first name emulates the datasets they want to use for the 
training/eval/vis, plus another for recovering interrupted processes without the need of continous updating 
of the initial training script.

**Synopsis:** `[dataset_name]_[train/eval/vis/recover_train].sh`
**Example:** `guitar_train.sh`

To launch them, do something like:
```bash
$ sh guitar_train.sh
```

Check each file for more details. We provided comments to explain the steps and the meaning of 
the configuration values. In general, you will need to choose:
* paths for the input tfrecords
* paths for output folders
* initial checkpoints (see the 
[model-zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) 
for details on pretrained models) or recovery checkpoints
* hyperparamenters like number of iterations (then number of epochs), batch size, GPUs used, crop sizes,
internal feature sizes and strides that resemble the model you are finetuning 

# Step 4: visualize results

`overlay.py` is a Python utility we provide to easily visualize the overlaid predicted label
of the results saved into the `exp/<split>/vis/` folder of your experiments. You can save the overlay
results, and decide how many of them you want to be displayed on the screen. Check:
```bash
$ python overlay.py --help
```

