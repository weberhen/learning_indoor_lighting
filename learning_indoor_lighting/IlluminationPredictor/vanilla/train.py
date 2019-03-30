# Henrique Weber, 2018

import sys
import os
from torch.utils import data
from learning_indoor_lighting.tools.utils import yaml_load, DictAsMember, load_from_file, setup_optimizer
from learning_indoor_lighting.tools.ldr_image import LDRImageHandler
from learning_indoor_lighting.tools.hdr_image import HDRImageHandler
from learning_indoor_lighting.IlluminationPredictor.vanilla.loader import IlluminationPredictorDataset
from learning_indoor_lighting.IlluminationPredictor.vanilla.callback import IlluminationPredictorCallback
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
    model = load_from_file(opt.network, 'IlluminationPredictorNet', opt)

    #
    #   Instantiate loaders
    #
    ldr_image_handler = LDRImageHandler(opt.ldr_mean_std)
    hdr_image_handler = HDRImageHandler(opt.hdr_mean_std, perform_scale_perturbation=False)
    train_dataset = IlluminationPredictorDataset(opt.data_path,
                                                 dataset_purpose='train',
                                                 configs=opt,
                                                 transform=ldr_image_handler)

    valid_dataset = IlluminationPredictorDataset(opt.data_path,
                                                 dataset_purpose='valid',
                                                 configs=opt,
                                                 transform=ldr_image_handler)

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
    optimizer = setup_optimizer(model, opt.optimizer, opt.learning_rate)

    train_loop_handler = TrainLoop(model, train_loader, valid_loader, optimizer, opt.backend,
                                   gradient_clip=opt.gradient_clip,
                                   use_tensorboard=opt.use_tensorboard)

    #
    #   Add callbacks
    #
    train_loop_handler.add_callback([IlluminationPredictorCallback(10, hdr_image_handler,
                                                                   ldr_image_handler,
                                                                   train_dataset,
                                                                   opt, model.ae)])

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