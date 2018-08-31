import os
from pytorch_toolbox.loop_callback_base import LoopCallbackBase
import numpy as np


class AutoEncoderCallback(LoopCallbackBase):
    def __init__(self, update_rate, hdr_image_handler, test_dataset, opt):
        self.update_rate = update_rate
        self.count = 0
        self.hdr_image_handler = hdr_image_handler
        self.from_index = test_dataset.from_index
        self.get_name = test_dataset.get_name
        self.opt = opt

    def batch(self, predictions, network_inputs, targets, info, is_train=True, tensorboard_logger=None):
        self.show_example(predictions, targets)
        if self.opt.save_estimation:
            _, filename = os.path.split(self.get_name(np.asarray(targets[0][0])[0]))
            save_path_filename = os.path.join(self.opt.output_path, filename)
            self.hdr_image_handler.save(predictions[0][0], save_path_filename, is_normalized=True)

    def epoch(self, epoch, loss, data_time, batch_time, is_train=True, tensorboard_logger=None):
        self.console_print(loss, data_time, batch_time, [], is_train)

    def show_example(self, prediction, target):
        self.hdr_image_handler.visualize(self.from_index(np.asarray(target[0][0])[0])[0][0], name='target',
                                         is_normalized=False, transpose_order=(2, 0, 1))
        self.hdr_image_handler.visualize(prediction[0][0], name='prediction', is_normalized=True,
                                         transpose_order=(2, 0, 1))
