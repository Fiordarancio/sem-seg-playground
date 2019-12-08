# Step 1: install Deeplab

Clone the whole `tensorflow/models` repo from [here](https://github.com/tensorflow/models).
Use the `requirements.txt` of this directory to install dependencies fast. In any case, 
check all the steps indicated in their [manual](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md).

# Step 2: dataset

Download your dataset and put it into the `datasets` folder. Substitute their `data_generator.py` with our
version to get the parameters applied on our `guitar` and `motive` datasets.

Modify `build_guitar_dataset.py` or `build_motive_dataset.py` in order to convert the prepared images
into *tfrecords* which is the file format the network actually wants as input.

# Step 3: launch

Here are some `_train/_eval/_vis` bash files, whose first name emulates the datasets they want to use
for the training. `_recover_train` files are useful to recover interrupted processes without the need of
continous updating of the initial `_train` file. 

We provided comments to explain the steps and the meaning of the configuration values. In general, 
you will need to choose:
* paths for the input tfrecords
* paths for output folders
* initial checkpoints (see the 
[model-zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) 
for details on pretrained models) or recovery checkpoints
* hyperparamenters like number of iterations (then number of epochs), batch size, GPUs used, crop sizes,
internal feature sizes and strides that resemble the model you are finetuning 

# Step 4: visualize results

`overlay.py` is a Python utility we provide to easily visualize the overlaid predicted label
of the results saved into the `exp/<split>/vis/` folder of your experiments.