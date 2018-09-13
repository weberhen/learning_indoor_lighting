from torchvision import transforms
from learning_indoor_lighting.tools.data_handler import DataHandler
from learning_indoor_lighting.tools.transformations import Scale, Log1p, NumpyImage2Tensor, Normalize, TonemapHDR
from learning_indoor_lighting.tools.utils import dataset_stats_load, save_hdr
import numpy as np


class HDRImageHandler(DataHandler):

    def __init__(self, mean_std_file, perform_scale_perturbation):
        super(HDRImageHandler, self).__init__()
        self.tn = TonemapHDR(gamma=1.4, percentile=90, max_mapping=0.8)
        self.dataset_mean, self.dataset_std = dataset_stats_load(mean_std_file)
        self.normalization_ops = [transforms.Compose([Scale(random_perturbation=perform_scale_perturbation),
                                                      Log1p(),
                                                      NumpyImage2Tensor(),
                                                      Normalize(self.dataset_mean,
                                                                self.dataset_std)])]

    def visualize(self, image, name, is_normalized, transpose_order=(0, 1, 2)):
        if is_normalized:
            image = self.unnormalize(image)
        hdr_unnormalized = np.clip(image, 0, image.max())
        _, ldr_unnormalized = self.tn(hdr_unnormalized, clip=True)
        ldr_unnormalized = ldr_unnormalized.transpose(transpose_order)
        self.vis.visualize(ldr_unnormalized, name)

    def save(self, image, filename, is_normalized):
        if is_normalized:
            image = self.unnormalize(image)
        hdr_unnormalized = np.clip(image, 0, image.max())
        save_hdr(hdr_unnormalized, filename)
