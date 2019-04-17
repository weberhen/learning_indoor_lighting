# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Labal
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import os
from torch.utils import data
from learning_indoor_lighting.AutoEncoder.loader import AutoEncoderDataset
from learning_indoor_lighting.tools.utils import yaml_load, DictAsMember, load_from_file, LatentVectorHandler
from learning_indoor_lighting.tools.hdr_image import HDRImageHandler
from learning_indoor_lighting.AutoEncoder.callback import AutoEncoderCallback
from pytorch_toolbox.train_loop import TrainLoop


def test(train_loop_handler):
    lvh = LatentVectorHandler()
    train_loop_handler.model.eval()

    for i, (data, target, info) in enumerate(train_loop_handler.valid_data):
        data, target = train_loop_handler.setup_loaded_data(data, target, train_loop_handler.backend)
        data_var, target_var = train_loop_handler.to_autograd(data, target, is_train=False)
        y_pred = train_loop_handler.model.encode(*data_var)
        lvh.append(y_pred, info['path'][0])
        y_pred = train_loop_handler.model.decode(*y_pred)
        if not isinstance(y_pred, tuple):
            y_pred = (y_pred,)

        for i, callback in enumerate(train_loop_handler.callbacks):
            callback.batch(y_pred, data, target, info, is_train=False,
                           tensorboard_logger=train_loop_handler.tensorboard_logger)

    _, file = os.path.split(opt.data_path)
    lvh.save('.', file)


if __name__ == '__main__':

    #
    #   Load configurations
    #
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "test.yml"
    opt = DictAsMember(yaml_load(config_path))

    #
    #   Instantiate models
    #
    model = load_from_file(opt.network, 'AutoEncoderNet')

    #
    #   Instantiate loaders
    #
    hdr_image_handler = HDRImageHandler(opt.mean_std, perform_scale_perturbation=False)

    test_dataset = AutoEncoderDataset(opt.data_path, transform=hdr_image_handler.normalization_ops)

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=0,
                                  pin_memory=True)

    #
    #   Instantiate the train loop
    #
    train_loop_handler = TrainLoop(model, None, test_loader, None, opt.backend)
    train_loop_handler.setup_checkpoint(opt.load_best,
                                        opt.load_last,
                                        opt.output_path,
                                        False)

    #
    #   Add callbacks
    #
    train_loop_handler.add_callback([AutoEncoderCallback(10, hdr_image_handler, test_dataset, opt)])

    #
    #   Test the model
    #
    test(train_loop_handler)
