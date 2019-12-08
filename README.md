# A semantic segmentation playground

This workspace collects a personal bunch of interesting projects 
about semantic segmentation, which is a computer vision task for doing
a "pixel-level" classification. This playground has been created in october 2019 
for carrying out joint developement by [EventLab](http://www.event-lab.org/).
Our goal is to test performances of different models in order to embed them
in Virtual Reality applications. We also aim to build an easy-to-use
workspace to train, evaluate and test these models over brand new data. 

When building a personal set of examples to be issued to the networks, it is
very important to ensure the correctness and coherence of the labels. In order
to help this, we publish a MATLAB based toolkit called [DataMonitor]()
which provides a number of interactive scripts for manual management of a given dataset.

The other projects listed refer to some State-Of-The-Art (SOTA) methods about 
the task of semantic segmentation, applied to many datasets 
(including [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/),
[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/),
[Cityscapes](https://www.cityscapes-dataset.com/) and 
[COCO](http://cocodataset.org/#home). Folders were 
forked by original authors, whose work we appreciate and 
<a href=#Acknowledgements>acknowledge</a>. Following the need of 
generalizing training and test over a our context-specific datasets,
we modified few things in their code or just followed their istructions.

**We exclusively put in this repository the files we actually modified or 
added brand new. In each subfolder you will find further guidelines to 
set up your own environment.**

The code is meant to be modular and reusable, so you should be able to
try over your own datasets without relevant difficulties just following
the examples provided in each project. If you find a bug, please report it
using the [issue tracker](https://github.com/Ilancia/sem-seg-playground/issues) 
or refer the original repository pages.

## Folder structure
* **DataMonitor**: MATLAB toolkit including interactive scripts to manage
datasets. It also allows MATLAB base training on some networks from scratch.
* **Deeplab, OCRnet, SPADE, Semseg-MIT**: folders with our own modifications of 
some projects to be forked (see below)

The presented models are mainly implemented using Python, Tensorsflow and
Pytorch. A related virtual environment has been created for each of them: 
this allows to manage required libraries without
internal conflicts. It is reccomended to export, at the end of each
`<venv_name>/bin/activate` the paths needed at installation time, instead
of exporting them at each new activation. For an example, see Deeplab's
<a href='models/research/deeplab/g3doc/installation.md'>Installation.</a><br>

## Dependencies

Into our directories we listed some minimal installation requirements you should 
follow in order to play with these models just as we did, in order to replicate
our results or master them for your own purposes.

Into the `requirements.txt` file all specific packages (including version) are 
listed. Download and install them by simply typing:
```bash
  $ pip install -r requirements.txt --force-reinstall
```
Please mind that sometimes versions can be conflicting. Exploit virtual 
environments to easily apply updates or start over without side effects if
something went wrong (e.g. when cached or legacy packages issue conflicts
by leaving unresolved trash).

To create a new virtual environment using `virtualenv`, type:
```bash
  $ virtualenv --system-site-packages -p python3 ./<venv_name>
```
to import packages already prepared on your system installation of Python.
See the [virtualenv](https://virtualenv.pypa.io/en/latest/) documentation for
issues and details.

## Referred projects

* **Deeplabv3+**: Tensorflow impletentation under `models/research/deeplab`.
                  *Associated venv:* `venv`. This repo provides a wide range of pretrained
                  model variants to work on all the mentioned datasets
* **HRNet**: Pytorch MIT CSAIL implementation bundles for some SOTA networks
             under `Semseg-mit`. The repo provides support for PSPNet, UperNet and HRNet
             basically trained over the ADE20K dataset (as a benchmark). *Associated* 
             *venv:* `mitenv`
* **OCRNet**: Pytorch implementation of the Object-Context-Recognition, which
              claims the top ranking over many datasets in 2019. Details into 
              `OCRNet.pytorch`. *Associated venv:* `ocrnenv`
* **SPADE**: NVIDIA project for a Generative Adversarial Network. *Associated venv:* `spadenv`

# Acknowledgements

GitHub pages of the original contributors: please refer to them for updates and cite their work.
* [Deeplab: DeepLabeling for Semantic Image Segmentation](https://github.com/tensorflow/models/tree/master/research/deeplab)
* [Semantic Segmentation on MIT ADE20K dataset in PyTorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)
* [OCNet: Object Context Network for Scene Parsing](https://github.com/PkuRainBow/OCNet.pytorch)
* [Semantic Image Synthesis with SPADE](https://github.com/NVlabs/SPADE)