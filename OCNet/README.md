# Quick start
This repository contains resources for training over new datasets the project published 
[here](https:///www.github.com/PkuRainBow/PCNet.pytorch). By first, clone their repository and add
(or substitute) the files provided in this repo at the proper destinations.

## Dependencies
Install dependencies using:
```bash
$ source ../ocnet-dedicated-venv/bin/activate
$ pip install -r requirements.txt [--force-reinstall]
```
We now point out some differences between the quick tutorial that the original repo offers. 
Our `requirements.txt` is already prepared according to them.

### Torch and Torchvision
In the official `README`, the authors claim that installing `torchvision` as is, is OK. However,
by now (December 2019) `torch==0.4.1` is not compatible with the latest `torchvision==0.4.2` which is 
installed automatically. So we need to specify version `torchvision==0.2.1`.

### Ninja
It is not mentioned in the requirement list too, but the program will blame you if you don't get it
installed. Just add `ninja`.

### OpenCV
Adding `opencv-python` should be enough, but if the code encounters problems, please find out the 
version needed. A walkaround is indeed necessary for Windows systems: follow the 
[docs](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html).

# Prepare your dataset
1.  Create a new folder dedicated to your samples, here or in the existing `./datasets/` (which is already
    prepared for Cityscapes). It should have the following structure
    ```
    mydataset
      |-- train_images
      |-- train_labels
      |-- eval_images
      |-- eval_labels
      |-- test_images
      |-\  list
    ```
2.  This project uses some python classes to load the information of the datasets. This is done through 
    some `.lst` files, that are, in practice, plain-text files contatining the path of each image and each
    correspondent label. The name of the list should be the one assignet to the dataset split that we are
    addressing. For example, `guitar_train.lst` can address the split `guitar_train` listing all images into
    `train_images` and `label_images`. Have a look into the `list` folder to see examples.
    To build these configuration files, you can run:
    ```bash
    $ python create_list.py --name [split_name] --img_dir [path_to_images] --lab_dir [path_to_labels]
    ```
    or alternatively, using the MATLAB function:
    ```
    createList('split_name', 'path/to/images', 'path/to/labels')
    ```
    In both cases, the label paths are optional, so that you can build a list with only images (e.g. for test splits)
    or with full image-labels examples.

3.  The classes for loading the information read on the lists must be been created using `citiscapes.py` or 
    `guitars.py` as examples. They are mainly parsers; the relevant modifications you will have to do is the
    map the classes that will be retrieved over the labels. Your dataset could contain more than the classes 
    you are really interested to train the network on, and there can be one or more labels to be ignored.
    Simply enumerate like that: `(list) [ [visible_class_index] : [actual_class_index | ignore_label]`. See
    `guitars.py` for more specific details!

4.  Add your new dataset names into the dictionary provided into `./__init__.py` as follows:
    ```python
    # datasets/__init__.py`
    datasets = {
      'cityscapes_train': CitySegmentationTrain,
      'cityscapes_test': CitySegmentationTest,
      'cityscapes_train_w_path': CitySegmentationTrainWpath,
      'split_name': MyDatasetSegmentationSplit,
      ... 
    }
    ```

# Prepare for launch!
We already prepared the bash files with relevant (and explained) parameters in `./guitars_resnet101_asp_oc.sh`.
We recommend using it as a starting point for scripts related to other datasets too!