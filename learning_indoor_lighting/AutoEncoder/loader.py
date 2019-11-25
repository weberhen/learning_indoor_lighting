# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Laval and Institute National d'Optique
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import numpy as np
import torch
from pytorch_toolbox.loader_base import LoaderBase
from learning_indoor_lighting.tools.utils import load_hdr


class AutoEncoderDataset(LoaderBase):

    def __init__(self, root, transform=[], target_transform=[]):
        self.images = self.make_dataset(root)
        super().__init__(root, transform, target_transform)

    def make_dataset(self, root):
        dir = os.path.expanduser(root)
        images = []
        files = [x for x in os.listdir(dir) if x.endswith('.exr')]
        for file in files:
            path = os.path.join(dir, file)
            images.append(path)

        return images

    def __len__(self):
        return len(self.images)

    def get_name(self, index):
        return self.images[index]

    def from_index(self, index):

        path = self.images[index]
        image = [load_hdr(path)]
        index = torch.from_numpy(np.array([index]))
        info = {'path': os.path.basename(os.path.normpath(path))}
        return image, index, info
