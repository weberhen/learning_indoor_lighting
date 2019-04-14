# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Labal
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import Imath
import OpenEXR
import ezexr
import os
import importlib
import yaml
import numpy as np


def dataset_stats_load(filename):
    """
    Loads the mean and standard deviation from a file with the following structure:
    Dataset MEAN R - G - B
    0.3505775317983936 0.33745820307121455 0.3108691100984408
    Dataset STD R - G - B
    0.2885629269999875 0.29015735209885285 0.2939203855287321
    :param filename: path to the file
    :return: mean (ndarray of shape 3) and standard deviation (ndarray of shape 3)
    """
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    mean = np.fromstring(lines[1], dtype='double', sep=' ')
    std = np.fromstring(lines[3], dtype='double', sep=' ')

    return mean, std


def find_images(root, extension='.exr'):
    """
    Returns the full path to all files with the given extension inside a folder
    :param root: str, folder where to look for files
    :param extension: str, extension of the files
    :return: list of paths
    """
    dir = os.path.expanduser(root)
    images_path = []
    files = [x for x in sorted(os.listdir(dir)) if x.endswith(extension)]
    for file in files:
        path = os.path.join(dir, file)
        images_path.append(path)

    return images_path


def load_hdr(path):
    """
    Loads an HDR file with RGB channels
    :param path: file location
    :return: HDR image
    """
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb_img_openexr = OpenEXR.InputFile(path)
    rgb_img = rgb_img_openexr.header()['dataWindow']
    size_img = (rgb_img.max.x - rgb_img.min.x + 1, rgb_img.max.y - rgb_img.min.y + 1)

    redstr = rgb_img_openexr.channel('R', pt)
    red = np.fromstring(redstr, dtype=np.float32)
    red.shape = (size_img[1], size_img[0])

    greenstr = rgb_img_openexr.channel('G', pt)
    green = np.fromstring(greenstr, dtype=np.float32)
    green.shape = (size_img[1], size_img[0])

    bluestr = rgb_img_openexr.channel('B', pt)
    blue = np.fromstring(bluestr, dtype=np.float32)
    blue.shape = (size_img[1], size_img[0])

    hdr_img = np.dstack((red, green, blue))

    return hdr_img


def load_hdr_multichannel(path):
    """
    Loads an HDR file with RGB, normal (3 channels) and depth (1 channel)
    :param path: file location
    :return: HDR image
    """
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb_img_openexr = OpenEXR.InputFile(path)
    rgb_img = rgb_img_openexr.header()['dataWindow']
    size_img = (rgb_img.max.x - rgb_img.min.x + 1, rgb_img.max.y - rgb_img.min.y + 1)

    # color
    redstr = rgb_img_openexr.channel('color.R', pt)
    red = np.fromstring(redstr, dtype=np.float32)
    red.shape = (size_img[1], size_img[0])

    greenstr = rgb_img_openexr.channel('color.G', pt)
    green = np.fromstring(greenstr, dtype=np.float32)
    green.shape = (size_img[1], size_img[0])

    bluestr = rgb_img_openexr.channel('color.B', pt)
    blue = np.fromstring(bluestr, dtype=np.float32)
    blue.shape = (size_img[1], size_img[0])

    color = np.dstack((red, green, blue))

    # normal
    normal_x_str = rgb_img_openexr.channel('normal.R', pt)
    normal_x = np.fromstring(normal_x_str, dtype=np.float32)
    normal_x.shape = (size_img[1], size_img[0])

    normal_y_str = rgb_img_openexr.channel('normal.G', pt)
    normal_y = np.fromstring(normal_y_str, dtype=np.float32)
    normal_y.shape = (size_img[1], size_img[0])

    normal_z_str = rgb_img_openexr.channel('normal.B', pt)
    normal_z = np.fromstring(normal_z_str, dtype=np.float32)
    normal_z.shape = (size_img[1], size_img[0])

    normal = np.dstack((normal_x, normal_y, normal_z))

    # depth
    depth_str = rgb_img_openexr.channel('distance.Y', pt)
    depth = np.fromstring(depth_str, dtype=np.float32)
    depth.shape = (size_img[1], size_img[0])

    return color, depth, normal


def save_hdr(_file, _filename):
    """
    Saves HDR image
    :param _file: HDR data
    :param _filename: str, full path and name
    :return: none
    """
    ezexr.imwrite(_filename, _file, pixeltype='HALF')


def save_estimation(_pred, _name, _path, _untransform_ops):
    """
    Remove normalization done on estimation, then save it
    :param _pred: HDR data
    :param _name: filename
    :param _path: path where to save
    :param _untransform_ops: list of operatios to unnormalize the data (bring it back to the RGB space)
    :return:
    """
    for i, transform in enumerate(_untransform_ops):
        pred_unnormalized = transform(_pred)
    pred_unnormalized = np.clip(pred_unnormalized, 0, pred_unnormalized.max())
    save_hdr(pred_unnormalized, "{}/{}".format(_path, _name))


def to_np(x):
    """
    Transforms x from Tensor to numpy
    :param x: Tensor data
    :return: numpy data
    """
    return x.data.cpu().numpy()


def yaml_load(filepath):
    """
    Load a yaml file
    :param filepath: path
    :return: dict with the yaml contents
    """
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        return data


class DictAsMember(dict):
    def __getattr__(self, name):
        """
        Transforms a dictionary into a class where attributes are the keys
        :param name: key of the dictionary
        :return: class with the key as member attribute
        """
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def load_from_file(module_name, class_name, *kargs):
    """
    Load a class inside a module
    :param module_name: str, module name
    :param class_name: str, class name
    :param kargs: any arguments to the class
    :return: class instance
    """
    class_inst = None

    py_mod = importlib.import_module(module_name)

    if hasattr(py_mod, class_name):
        class_inst = getattr(py_mod, class_name)(*kargs)

    assert class_inst

    return class_inst


def check_nans(x):
    """
    Check for NaNs and infinities
    :param x: Tensor
    :return: none
    """
    nans = np.sum(np.isnan(x.cpu().data.numpy()))
    infs = np.sum(np.isinf(x.cpu().data.numpy()))
    if nans > 0:
        print("There is {} NaN at the output layer".format(nans))
    if infs > 0:
        print("There is {} infinite values at the output layer".format(infs))


def setup_optimizer(model, chosen_optimizer, learning_rate):
    """
    Configures optimizer for training
    :param model: pytorch model to be trained
    :param chosen_optimizer: str, name of the optimizer
    :param learning_rate: learning rate
    :return: optimizer
    """
    from torch import optim
    if chosen_optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif chosen_optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        print("Optimizer not configures")
        exit(0)

    return optimizer

