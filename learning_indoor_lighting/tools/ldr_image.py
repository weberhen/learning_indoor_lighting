from torchvision import transforms
import numpy as np
from illuminnet.tools.data_handler import DataHandler
from illuminnet.tools.transformations import ToFloat, NumpyImage2Tensor, Normalize
from illuminnet.tools.utils import dataset_stats_load


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
