# Henrique Weber, 2018

import sys
from torch.utils import data
from learning_indoor_lighting.AutoEncoder.vanilla.loader import AutoEncoderDataset
from learning_indoor_lighting.tools.utils import yaml_load, DictAsMember, load_from_file
from learning_indoor_lighting.tools.hdr_image import HDRImageHandler
from learning_indoor_lighting.AutoEncoder.vanilla.callback import AutoEncoderCallback
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
    train_loop_handler.test()

