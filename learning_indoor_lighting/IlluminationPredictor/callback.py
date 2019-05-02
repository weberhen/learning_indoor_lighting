# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Laval and Institute National d'Optique
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from pytorch_toolbox.loop_callback_base import LoopCallbackBase
from learning_indoor_lighting.tools.utils import to_np
from torch.autograd import Variable


class IlluminationPredictorCallback(LoopCallbackBase):
    def __init__(self, update_rate, hdr_image_handler, ldr_image_handler, test_dataset, opt, ae_model, testing_loader=False):
        self.update_rate = update_rate
        self.count = 0
        self.hdr_image_handler = hdr_image_handler
        self.ldr_image_handler = ldr_image_handler
        self.from_index = test_dataset.from_index
        self.get_name = test_dataset.get_name
        self.opt = opt
        self.ae_model = ae_model
        self.ae_model.eval()
        self.testing_loader = testing_loader

    def batch(self, predictions, network_inputs, targets, info, is_train=True, tensorboard_logger=None):
        self.ae_model.eval()
        self.show_example(predictions, targets, info)
        if self.opt.save_estimation:
            _, filename = os.path.split(info['name'][0])
            save_path_filename = os.path.join(self.opt.output_path, filename)
            latent_vector = predictions[0][0]
            pred_image = self.ae_model.decode(latent_vector)
            self.hdr_image_handler.save(pred_image, save_path_filename, is_normalized=True)

    def epoch(self, epoch, loss, data_time, batch_time, is_train=True, tensorboard_logger=None):
        self.console_print(loss, data_time, batch_time, [], is_train)

    def show_example(self, prediction, target, info):
        self.ae_model.eval()
        if not self.testing_loader:
            targ_image = self.ae_model.decode(Variable(target[0][0]).to(self.opt.backend))
            self.hdr_image_handler.visualize(targ_image, name='target', is_normalized=True, transpose_order=(2,0,1))
        self.ldr_image_handler.visualize(to_np(Variable(info['ldr_img'][0])),
                                         name='input', is_normalized=False)
        latent_vector = prediction[0][0]
        pred_image = self.ae_model.decode(latent_vector)
        self.hdr_image_handler.visualize(pred_image, name='prediction', is_normalized=True, transpose_order=(2,0,1))
