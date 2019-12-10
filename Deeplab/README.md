# Step 1: install Deeplab

Clone the whole `tensorflow/models` repo from [here](https://github.com/tensorflow/models).
Create a virtual environment and use the `requirements.txt` of this directory to install dependencies fast. 
In any case, check all the steps indicated in their 
[manual](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md).

# Step 2: dataset

Download your dataset and put it into deeplab's `datasets` folder. Substitute their `data_generator.py` with our
version to get the parameters applied on our `guitar` and `motive` datasets. You can check the folders that we
uploaded here.

Modify `build_guitar_dataset.py` or `build_motive_dataset.py` in order to convert the prepared images
into *tfrecords* which is the file format the network actually wants as input.

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
```python
$ python overlay.py --help
```