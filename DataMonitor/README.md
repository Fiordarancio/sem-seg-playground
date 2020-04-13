# GUITAR SEGMENTER - version 1.1.0 - december 2019

*(NEED TO ADD: UNET SETUP, GAN SETUP, EVENT DATASET SETUP)*

Bundle of programs to manage sets of examples for semantic segmentation
oriented networks. Many basic tools for creating and checking datasets
are here provided, together with some modified MATLAB tutorials to train
simple networks. You can:
* apply pre-processing on images
* check if a group of labels is correct or not
* use a pretrained network to segment new images (inference only
* perform a new training/finetuning of a network backbone with custom training options or a custom dataset

Implemented network backbones are:
* [DeepLabV3+](https://arxiv.org/pdf/1802.02611.pdf), built over ResNet-18
* [SegNet](https://arxiv.org/pdf/1511.00561.pdf)

## 1. Requirements

This project has been developed using MATLAB 2019b. Compatibility with
previous versions of the software is not granted.
Sources may require the following packages installed:
* Computer Vision Toolbox
* Deep Learning Toolbox
* Deep Learning Toolbox Model for VGG-16 Network
* Deep Learning Toolbox Model for Resnet-18 Network
* Global Optimization Toolbox
* GPU Coder
* Optimization Toolbox
* Parallel Computing Toolbox

## 2. Resources

Available folders are arranged in this way
```
Guitar_ilaria
  |- dataset
      |- ConcertDataset
      |- ImagenetDataset
      |- OpenImageDatasetV5
      |- templateColStd
      |- ZhinhanDataset
      |- ...
  |- fun
  |- guitarNetCheckpoint
  |- guitarOutputTesLabels
  |- guitarResults
  |- temp_camvid
```
In the main folder, we point out 4 scripts:
* **Guitar_deeplab**: script for setting up a training or testing environment
  which uses the DeepLabV3 structure to deploy a network able to
  recognize a guitar over a given dataset. Through this script it is
  possible to launch a new training with given dataset and options or to
  exploit an existing pre-trained architecture in order to test
  performances over a new dataset
* **Guitar_segnet**: pretty like the same as before, now using SegnNet, which
  is based on UNet layers (Encoder*Decoder)
* **Example_deeplab_camvid**: script for dowloading and training/testing a
  DeepLabV3-based network set and pretrained on CamVid dataset. use it to
  check how things work, according with [MATLAB`s official tutorial]
  (https://es.mathworks.com/help/vision/examples/semantic-segmentation-using-deep-learning.html)

The `dataset` folder contains all the downloaded and processed datasets
used until now. Moreover, it presents scripts that can be used for 
preparing new custom datasets, as explained in chapter 3. Each set
contains one `images` and one `labels` folder so, when adding new data,
please keep this convention.

Into this folder you can also find `templatesColstd`, which contains
images with nice and diverse color arrangements, in order to perform a
various color standardization, as explained in chapter 3.2.

The `fun` folder contains the functions used by the scripts available in
the main one.

`guitarNetcheckpoint`, `guitarOutputTestLabels` and `guitarResults`
contain log files and results produced by the main scripts described
above.

The `temp_camvid` folder contains resources used by the tutorial
`Example_deeplab_camvid.m`

In general, snippets that can be modified by the user for plug & play
purposes are marked with line separations and BEGIN USER CODE - END USER
CODE labels. We hope this would make interactions easier.

### 2.1 MoTIVE Dataset

*Section to be updated*

This is the dataset that we are building up into the EventLab. It should
be called ConcertDataset or with another more specific name, since it
covers about 35 categories of generic rock concert scenarios. In order to
be more generic and to allow the final MoTIVE application to recontruct
many kinds of concerts, it could be necessary to add some of them. 
There is NO open source dataset of musical instruments available on the
internet, except the ImageNet one. A contribution in this field could be
very useful.

In the folder `MoTIVEDataset`, the script `analyzeExamples.m` provides a
tool to visually inspect if the labels are correct or not, and to act
over it. The script scrolls the images showing the categories and the
instances separately, and also saves them in different folders.

## 3. Prepare the dataset
### 3.1 Requirements
Both DeepLabV3 and SegnNet require images and labels that mandatorily fit
the size (and aspect ratio as consequence) expected by their input layer. 
```
  | NETWORK   | SIZE (WIDTH X HEIGHT) | CHANNELS  |
  | ----------|-----------------------|---------- |
  | DeepLabV3 | 960 x 720 uint8       | 3 RGB     |
  | SegNet    | 480 x 360 uint8       | 3 RGB     |
```

You can put your images and labels into the folder `dataset`. Some
utilities are provided to build up a dataset suited for using/training
your network. 

Note that the number of images of the ground truth MUST be equal to the 
number of labels. Moreover, images must be uniquely associated to their
labels by having the same name (file extension may differ). Name matching
is not mandatory and does not strictly generate errors; however, of 
course, this will generate wrong associations and bad performaces

### 3.2 Build your dataset
Into the `dataset` folder, copy your images and put them in a new
directory of custom name. Copy your labels as well (if needed for a new 
training) using the same procedure: divide files into `images` and
`labels`.

You can add your data to the existing dataset by properly modifying the
script `prepareEnhancedDataset.m`: just add the name of the newly created
folder to the list of datasets and run then the script. 

To fulfill the aforementioned requisites, you have to properly set up  
data, and their folders as well. There are two main scripts for this:
`prepareExamplesDeepLab.m` and `prepareExamplesSegnet.m`. In each of
them, the user can choose which type of operation has to be applied on
the original dataset.

Example:  we want to add myPersonalDataset to the overall dataset; then, 
          we want to prepare examples for DeepLab, using padding & crop 
          and color stardization methods

1. add `myPersonalDataset` into the folder `dataset`
2. be sure that data into `myPersonalDataset` are grouped into two
   folders, called `images` and `labels` respectively
3. be sure that the number of elements in these folders matches, as well
   as file names
4. open `prepareEnhancedDataset.m` and insert "myPersonalDataset" into
   the dataset list
5. launch `prepareEnhancedDataset.m`
6. open `prepareExamplesDeeplab.m` and
  2. change the target paths (input folders)
  3. select the augmentations to apply setting the relative flags 
4. launch `prepareExamplesDeeplab.m`

If you just want to use `myPersonalDataset` only, just skip steps 4. and 5.

## 4. Use a pretrained network
### 4.1 Description
The folder `guitarResults` contains workspaces and log data which follow
this terminology:
```bash
  net_<date-of-train>__<hour-24h-of-train>_<name-of-network>
```

The workspaces include the network to be used directly for new detections
and also test cases (`net` variable), together with accuracy evaluations.
Figures follow a similar nomenclature: the ones which last with `_tp` 
illustrate the training process plot of the relative training.

### 4.2 How to
1. import your dataset as explained in chapter 3.2.
2. load the desired workspace from `guitarResults`
3. update paths into `pathSetup.m` with the name of the folders in which
   you put your test images (for Deeplab ones, we recommend using 
	`imagesDeeplab` and `labelsDeeplab`, and the same for Segnet)
4. launch `Guitar_test_deeplab.m` or `Guitar_test_segnet.m`

### 4.3 Train again your network

If you want to build up your network from the present templates, you have
to go deeper in the update of `Guitar_deeplab.m` and `Guitar_segnet.m`.
Dataset`s setup just remains the same, but instead you are going to vary
the properties of you network. Mainly, you can tune the hyperparameters
to be applied during the training, for example:
* mini batch size
* max number of epochs
* training options (see MATLAB`s documentation for further information)

Our scripts have a setup already coded, so just modify the lines.
Remember that at the end of each training all the workspace is saved as
explained in chapter 4.1. so that it is possible to compare network
results considering avery made choice.

## Citation
EventLab, Universitad de Barcelona, ES
