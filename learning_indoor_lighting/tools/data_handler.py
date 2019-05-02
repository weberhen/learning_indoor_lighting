# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Laval and Institute National d'Optique
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from learning_indoor_lighting.tools.visdom_logger import VisdomLogger


class DataHandler:

    def __init__(self):
        self.normalization_ops = []
        self.vis = VisdomLogger()

    def get_normalization_ops(self):
        return self.normalization_ops

    def normalize(self, image):
        image_normalized = np.zeros(image.shape)
        for i, transforms in enumerate(self.normalization_ops):
            for j, transform in enumerate(transforms.transforms):
                image_normalized = transform(image)
                image = image_normalized
        return image_normalized

    def unnormalize(self, image):
        image_unnormalized = np.zeros(image.shape)
        for i, transforms in enumerate(self.normalization_ops):
            for j, transform in enumerate(reversed(transforms.transforms)):
                image_unnormalized = transform.undo(image)
                image = image_unnormalized
        return image_unnormalized
