##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## Modified by: RainbowSecret 
## Latest update by: Ilaria Tono
## EventLab, Spain, 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import cv2
import pdb
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from PIL import Image, ImageOps, ImageFilter
import random
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

class GuitarSegmentationTrain(data.Dataset):
  # This is the class for loading and managing the training set. Actually, the same 
  # procedures are applied on the evaluation set, so in the declaration the evaluation
  # dataset will be loaded using this class too.

    # We follow the crop size is like the one of Deeplab. 
    # See how performances change while using or not the augmentation.
    def __init__(self, root, list_path, max_iters=None, crop_size=(513, 513),
        scale=True, mirror=True, ignore_label=255, use_aug=False, network="renset101"):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label     
        self.is_mirror = mirror
        self.use_aug = use_aug
        self.network = network
        # Open the list file and read the examples IDs.
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })
        # List the classes that must be recognized. In the example of guitars, they merged some
        # classes into others (like we did also for CamVid, Pascal and ADE20k) following this
        # syntax: [num of label]: [num of label we want | ignore_label]. In our case, we get all but
        # the Background class which is going to be ignored
        self.id_to_trainid = {0: ignore_label, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10,
                              12:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:17, 19:18, 20:19, 21:20, 
                              22:21, 23:22, 24:23, 25:24, 26:25, 27:26, 28:27, 29:28, 30:29, 31:30,
                              32:31, 33:32, 34:33, 35:34}

        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.use_aug: # the augmented data gt label map has been transformed
            label = label
        else:
            label = self.id2trainId(label)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        

        if self.network == "resnet101":
            mean = (102.9801, 115.9465, 122.7717)
            image = image[:,:,::-1]
            image -= mean
        elif self.network == "mobilenetv2":
            mean = (0.485, 0.456, 0.406)
            var = (0.229, 0.224, 0.225)
            # print("network: {}, mean: {}, var: {}".format(self.network, mean, var))
            image = image[:,:,::-1]
            image /= 255
            image -= mean   
            image /= var
        elif self.network == "wide_resnet38":
            mean = (0.41738699, 0.45732192, 0.46886091)
            var = (0.25685097, 0.26509955, 0.29067996)
            image = image[:,:,::-1]
            image /= 255
            image -= mean   
            image /= var

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(), label.copy(), np.array(size), name


class GuitarSegmentationTest(data.Dataset):
  # The test set does not need labels.
    def __init__(self, root, list_path, max_iters=None, crop_size=(513, 513),
        scale=True, mirror=True, ignore_label=255, network=None):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.network = network  
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            image_path = item
            name = osp.splitext(osp.basename(image_path[0]))[0]
            img_file = osp.join(self.root, image_path[0])
            self.files.append({
                "img": img_file,
                "name": name
            })
        self.id_to_trainid = {0: ignore_label, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10,
                              12:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:17, 19:18, 20:19, 21:20, 
                              22:21, 23:22, 24:23, 25:24, 26:25, 27:26, 28:27, 29:28, 30:29, 31:30,
                              32:31, 33:32, 34:33, 35:34}

        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)

        if self.network == "resnet101":
            mean = (102.9801, 115.9465, 122.7717)
            image = image[:,:,::-1]
            image -= mean
        elif self.network == "mobilenetv2":
            mean = (0.485, 0.456, 0.406)
            var = (0.229, 0.224, 0.225)
            image = image[:,:,::-1]
            image /= 255
            image -= mean   
            image /= var
        elif self.network == "wide_resnet38":
            mean = (0.41738699, 0.45732192, 0.46886091)
            var = (0.25685097, 0.26509955, 0.29067996)
            image = image[:,:,::-1]
            image /= 255
            image -= mean   
            image /= var

        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
        else:
            img_pad = image

        img_h, img_w, _ = img_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name

# # This is useless for our purposes...?
# if __name__ == '__main__':
#     dst = GuitarSegmentationTrain("./guitars/", "./list/guitars/train.lst", crop_size=(513, 513))
#     trainloader = data.DataLoader(dst, batch_size=1, num_workers=0)

#     with open("./list/guitars/train.lst") as f:
#         train_list = f.readlines()
#     train_list = [x.strip() for x in train_list] 

#     f_w = open("./list/guitars/bus_truck_train.lst", "w")
#     cnt = np.zeros(19)
#     for i, data in enumerate(trainloader):
#         freq = np.zeros(19)
#         imgs, labels, _, _ = data
#         n, h, w = labels.shape
#         print('prcessing {}th images ...'.format(i))
#         for k in [14, 15, 16]:
#             mask = (labels[:, :, :] == k)
#             freq[k] += torch.sum(mask)
#             if freq[k] > 200:
#                  f_w.writelines(train_list[i]+"\n")
#                  cnt[k] += 1
#                  print('# images of class {}: {}'.format(k, cnt[k]))
#     print(cnt)