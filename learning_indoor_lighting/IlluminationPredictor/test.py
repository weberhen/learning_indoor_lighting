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
import torch
from torch.utils import data
from learning_indoor_lighting.tools.utils import yaml_load, DictAsMember, load_from_file
from learning_indoor_lighting.tools.ldr_image import LDRImageHandler
from learning_indoor_lighting.tools.hdr_image import HDRImageHandler
from learning_indoor_lighting.IlluminationPredictor.loader import IlluminationPredictorDataset
from learning_indoor_lighting.IlluminationPredictor.callback import IlluminationPredictorCallback
from pytorch_toolbox.train_loop import TrainLoop


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
    model = load_from_file(opt.network, 'IlluminationPredictorNet', opt)

    #
    #   Create output folder
    #
    command = "mkdir -p {}".format(opt.output_path)
    os.system(command)

    #
    #   Instantiate loaders
    #
    ldr_image_handler = LDRImageHandler(opt.ldr_mean_std)
    hdr_image_handler = HDRImageHandler(opt.hdr_mean_std, perform_scale_perturbation=False)
    test_dataset = IlluminationPredictorDataset(opt.data_path,
                                                configs=opt,
                                                transform=ldr_image_handler,
                                                dataset_purpose='test')

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=opt.workers,
                                  pin_memory=opt.use_shared_memory)

    #
    #   Instantiate the train loop
    #
    train_loop_handler = TrainLoop(model, None, test_loader, None, opt.backend,
                                   gradient_clip=False,
                                   use_tensorboard=False)
    train_loop_handler.setup_checkpoint(opt.load_best, opt.load_last, opt.model_path, False)

    #
    #   Add callbacks
    #
    train_loop_handler.add_callback([IlluminationPredictorCallback(1, hdr_image_handler, ldr_image_handler,
                                                                   test_dataset, opt, model.ae, testing_loader=True)])

    #
    #   Test the model
    #
    train_loop_handler.test()
