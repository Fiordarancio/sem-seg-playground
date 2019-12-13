from .cityscapes import CitySegmentationTrain, CitySegmentationTest, CitySegmentationTrainWpath
from .guitars import GuitarSegmentationTrain, GuitarSegmentationTest

# Here is a dictionary of the datasets that can be loaded. You can think of each entry
# as a (cumbersome) alternative to Deeplab's split
datasets = {
	'cityscapes_train': CitySegmentationTrain,
	'cityscapes_test': CitySegmentationTest,
	'cityscapes_train_w_path': CitySegmentationTrainWpath,
	'guitars_train': GuitarSegmentationTrain,
	'guitars_eval': GuitarSegmentationTrain,
	'guitars_vis': GuitarSegmentationTest,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)