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
import torch
from numpy import inf
import numpy as np
from pytorch_toolbox.loader_base import LoaderBase
from learning_indoor_lighting.tools.utils import find_images, load_hdr_multichannel
from learning_indoor_lighting.tools.transformations import TonemapHDR


class IlluminationPredictorDataset(LoaderBase):

    def __init__(self, root_dir, configs, dataset_purpose, transform=[], target_transform=[]):
        self.configs = configs
        self.transform = transform
        self.dataset_purpose = dataset_purpose
        self.tonemap_hdr = TonemapHDR(gamma=1, percentile=90, max_mapping=.8)
        super().__init__(root_dir, transform.normalization_ops, target_transform)

        if self.dataset_purpose != 'test':
            f_names = open('{}/train_z_names.txt'.format(configs.ae_path), 'r')
            f_values = open('{}/train_z_values.txt'.format(configs.ae_path), 'r')
            self.dict_envmap_z = dict(zip(f_names.read().splitlines(), f_values.read().splitlines()))

            f_names = open('{}/valid_z_names.txt'.format(configs.ae_path), 'r')
            f_values = open('{}/valid_z_values.txt'.format(configs.ae_path), 'r')
            self.dict_envmap_z.update(dict(zip(f_names.read().splitlines(), f_values.read().splitlines())))

    def make_dataset(self, dir):
        images = []
        if self.dataset_purpose != 'test':
            images.extend(find_images(os.path.join(dir, self.dataset_purpose)))
        else:
            images.extend(find_images(dir))
        import random
        random.shuffle(images)
        return images

    def get_name(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)

    def from_index(self, index):
        rgb_path = self.imgs[index]

        path, file = os.path.split(rgb_path)
        multichannel_filename = rgb_path

        original_rgb_img, _, normals = load_hdr_multichannel(multichannel_filename)

        normals = np.flip(normals, axis=1).copy()
        original_rgb_img = np.flip(original_rgb_img, axis=1).copy()

        alphas, rgb_img = self.tonemap_hdr(original_rgb_img, clip=True)
        rgb_img = [rgb_img]
        normals[(abs(normals) == inf).any(axis=2)] = 0
        mask_normals = normals == 0
        assert not np.isinf(normals).any()

        if normals.max() != 0:
            normals = (normals - normals.min()) / (normals.max() - normals.min())

        if normals.max() != 1:
            normals = np.zeros(normals.shape)
            print('ERROR')
        try:
            assert normals.max() == 1
            assert normals.min() == 0
        except AssertionError:
            print(normals.max())
            print(rgb_path)

        normals[mask_normals] = 0
        normals = normals.transpose(2, 0, 1)
        rgb_img[0] = rgb_img[0].transpose(2, 0, 1)

        data = np.concatenate((rgb_img[0], normals))
        data = data.transpose(1,2,0)

        if self.dataset_purpose != 'test':
            target = np.expand_dims(np.fromstring(self.dict_envmap_z[file.rsplit('_m', 1)[0]+'.exr'], dtype=np.float32,
                                                  sep=' '), axis=0)
            target_latent_vector = target.squeeze()
            # target_latent_vector = torch.from_numpy(target_latent_vector).to(self.configs.backend)
        else:
            target_latent_vector = []

        target = {'name': file.replace('_multichannel_0_0', ''),
                  'alpha': alphas,
                  'ldr_img': rgb_img[0],
                  }

        return data, target_latent_vector, target
