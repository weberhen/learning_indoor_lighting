import numpy as np
from illuminnet.tools.visdom_logger import VisdomLogger


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
