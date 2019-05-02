# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Laval and Institute National d'Optique
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import os
from torch import optim
from torch.utils import data
from learning_indoor_lighting.AutoEncoder.loader import AutoEncoderDataset
from learning_indoor_lighting.tools.utils import yaml_load, DictAsMember, load_from_file
from learning_indoor_lighting.tools.hdr_image import HDRImageHandler
from learning_indoor_lighting.AutoEncoder.callback import AutoEncoderCallback
from pytorch_toolbox.train_loop import TrainLoop

if __name__ == '__main__':

    #
    #   Load configurations
    #
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "train.yml"
    opt = DictAsMember(yaml_load(config_path))

    #
    #   Instantiate models
    #
    model = load_from_file(opt.network, 'AutoEncoderNet')

    #
    #   Instantiate loaders
    #
    hdr_image_handler = HDRImageHandler(opt.mean_std, perform_scale_perturbation=True)
    train_dataset = AutoEncoderDataset(os.path.join(opt.data_path, "train"),
                                       transform=hdr_image_handler.normalization_ops)

    valid_dataset = AutoEncoderDataset(os.path.join(opt.data_path, "valid"),
                                       transform=hdr_image_handler.normalization_ops)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=opt.batch_size,
                                   num_workers=opt.workers,
                                   pin_memory=opt.use_shared_memory,
                                   drop_last=True)

    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=opt.batch_size,
                                   num_workers=opt.workers,
                                   pin_memory=opt.use_shared_memory)
    #
    #   Instantiate the train loop
    #
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)

    train_loop_handler = TrainLoop(model, train_loader, valid_loader, optimizer, opt.backend,
                                   gradient_clip=opt.gradient_clip,
                                   use_tensorboard=opt.use_tensorboard)

    #
    #   Add callbacks
    #
    train_loop_handler.add_callback([AutoEncoderCallback(10, hdr_image_handler, train_dataset, opt)])

    #
    #   Train the model
    #
    train_loop_handler.loop(opt.epochs,
                            opt.model_path,
                            load_best_checkpoint=opt.load_best,
                            save_best_checkpoint=True,
                            save_all_checkpoints=False,
                            forget_best_prec=opt.forget_best_prec
                            )

