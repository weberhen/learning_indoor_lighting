# Henrique Weber, 2018
# Utility functions

import Imath
import OpenEXR
import os
import importlib
from learning_indoor_lighting.tools.transformations import *
from learning_indoor_lighting.tools.visdom_logger import VisdomLogger
import yaml


def dataset_stats_load(filename):
    # given the complete path and name of the file, loads the statistics
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    mean = np.fromstring(lines[1], dtype='double', sep=' ')
    std = np.fromstring(lines[3], dtype='double', sep=' ')

    return mean, std


def find_images(root, extension='.exr'):
    dir = os.path.expanduser(root)
    images_path = []
    files = [x for x in sorted(os.listdir(dir)) if x.endswith(extension)]
    for file in files:
        path = os.path.join(dir, file)
        images_path.append(path)

    return images_path


def load_hdr(path):
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

    #depth
    depth_str = rgb_img_openexr.channel('distance.Y', pt)
    depth = np.fromstring(depth_str, dtype=np.float32)
    depth.shape = (size_img[1], size_img[0])

    return color, depth, normal


def save_hdr(_file, _filename):
    import ezexr
    ezexr.imwrite(_filename, _file, pixeltype='HALF')


def save_estimation(_pred, _name, _path, _untransform_ops):
    for i, transform in enumerate(_untransform_ops):
        pred_unnormalized = transform(_pred)
    pred_unnormalized = np.clip(pred_unnormalized, 0, pred_unnormalized.max())
    save_hdr(pred_unnormalized, "{}/{}".format(_path, _name))


def to_np(x):
    return x.data.cpu().numpy()


def log_tensorboard(_model, epoch, tensorboard_logger):
    for tag, value in _model.named_parameters():
        tag = tag.replace('.', '/')
        tensorboard_logger.histo_summary(tag, to_np(value), epoch + 1)
        try:
            tensorboard_logger.histo_summary(tag + '/grad', to_np(value.grad), epoch + 1)
        except AttributeError:
            tensorboard_logger.histo_summary(tag + '/grad', np.asarray([0]), epoch + 1)


def yaml_load(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor)
        return data


def find_images(root, extension='.exr'):
    dir = os.path.expanduser(root)
    images_path = []
    files = [x for x in sorted(os.listdir(dir)) if x.endswith(extension)]
    for file in files:
        path = os.path.join(dir, file)
        images_path.append(path)

    return images_path


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def load_from_file(module_name, class_name, *kargs):
    class_inst = None

    py_mod = importlib.import_module(module_name)

    if hasattr(py_mod, class_name):
        class_inst = getattr(py_mod, class_name)(*kargs)

    assert class_inst

    return class_inst


def check_nans(x):
    # Check for NaNs and infinities
    nans = np.sum(np.isnan(x.cpu().data.numpy()))
    infs = np.sum(np.isinf(x.cpu().data.numpy()))
    if nans > 0:
        print("There is {} NaN at the output layer".format(nans))
    if infs > 0:
        print("There is {} infinite values at the output layer".format(infs))


def setup_optimizer(model, chosen_optimizer , learning_rate):
    from torch import optim
    if chosen_optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    return optimizer