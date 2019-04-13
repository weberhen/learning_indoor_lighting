# Learning to Estimate Indoor Lighting from 3D Objects
# http://vision.gel.ulaval.ca/~jflalonde/projects/illumPredict/index.html
# Henrique Weber, 2018

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

        rgb_path = self.images[index]
        image = [load_hdr(rgb_path)]
        index = torch.from_numpy(np.array([index]))
        return image, index, []
