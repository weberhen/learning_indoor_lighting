# Learning to Estimate Indoor Lighting from 3D Objects
# http://vision.gel.ulaval.ca/~jflalonde/projects/illumPredict/index.html
# Henrique Weber, 2018

import torch.nn.functional as f
import torch.nn as nn
import torch
from torch.autograd import Variable
from pytorch_toolbox.network_base import NetworkBase
from learning_indoor_lighting.tools.utils import check_nans
from learning_indoor_lighting.AutoEncoder.vanilla.net import AutoEncoderNet


class IlluminationPredictorNet(NetworkBase):
    def __init__(self, configs):
        super(IlluminationPredictorNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, 5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, 3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, 3, stride=2)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(4608, 256)
        self.conv5_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)

        self.configs = configs
        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss()

        # load autoencoder to decode the latent vector produced by the IlluminationPredictorNet
        self.ae = AutoEncoderNet()
        if self.configs.train:
            filename = 'model_best.pth.tar'
            self.ae.load_state_dict(torch.load('{}/{}'.format(configs.ae_path, filename), map_location=configs.backend)
                                    ['state_dict'])

        self.ae.eval()
        self.ae.to(configs.backend)

    def forward(self, x):

        x = f.relu(self.conv1(x))
        x = self.conv2_bn(self.relu(self.conv2(x)))
        x = self.conv3_bn(self.relu(self.conv3(x)))
        x = self.conv4_bn(self.relu(self.conv4(x)))
        x = x.view(-1, 4608)
        x = self.conv5_bn(self.fc1(x))
        x = self.fc2(x)

        check_nans(x)

        return x

    def loss(self, predictions, targets):
        lz = self.criterion(predictions[0], Variable(targets[0], requires_grad=False).cuda())
        return lz

    @staticmethod
    def get_name_model():
        return 'model_best.pth.tar'
