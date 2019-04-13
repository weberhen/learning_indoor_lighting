# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Labal
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from torchvision import transforms
from learning_indoor_lighting.tools.data_handler import DataHandler
from learning_indoor_lighting.tools.transformations import ToFloat, NumpyImage2Tensor, Normalize
from learning_indoor_lighting.tools.utils import dataset_stats_load


class LDRImageHandler(DataHandler):

    def __init__(self, mean_std_file):
        super(LDRImageHandler, self).__init__()

        self.dataset_mean, self.dataset_std = dataset_stats_load(mean_std_file)
        self.normalization_ops = [transforms.Compose([ToFloat(),
                                                      NumpyImage2Tensor(),
                                                      Normalize(self.dataset_mean,
                                                                self.dataset_std)])]

    def visualize(self, image, name, is_normalized):
        if is_normalized:
            image = self.unnormalize(image)
        self.vis.visualize(image, name)

    def save(self, image, name, is_normalized):
        pass
