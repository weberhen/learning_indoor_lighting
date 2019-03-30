import torch.nn.functional as f
from tools.utils import *
from learning_indoor_lighting.AutoEncoder.vanilla.net import AutoEncoderNet
from pytorch_toolbox.network_base import NetworkBase

# python 3 confusing imports :(
from .unet_parts import *


class IlluminationPredictorNet(NetworkBase):
    def __init__(self, configs):
        super(IlluminationPredictorNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # encoder
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
        self.drop = nn.Dropout2d(p=.2)
        self.fc2 = nn.Linear(256, 128)
        self.loaded_ae = False

        self.configs = configs

        self.criterion = nn.MSELoss()

        filename = 'model_best.pth.tar'
        self.ae = AutoEncoderNet()
        self.ae.load_state_dict(torch.load('{}/{}'.format(configs.ae_path, filename))['state_dict'])

        self.ae.eval()
        self.ae.cuda()

    def forward(self, x):

        x = f.relu(self.conv1(x))
        x = self.conv2_bn(self.relu(self.conv2(x)))
        x = self.conv3_bn(self.relu(self.conv3(x)))
        x = self.conv4_bn(self.relu(self.conv4(x)))
        x = x.view(-1, 4608)
        x = self.conv5_bn(self.fc1(x))
        x = self.fc2(x)

        # Check for NaNs and infinities
        nans = np.sum(np.isnan(x.cpu().data.numpy()))
        infs = np.sum(np.isinf(x.cpu().data.numpy()))
        if nans > 0:
            print("There is {} NaN at the output layer".format(nans))
        if infs > 0:
            print("There is {} infinite values at the output layer".format(infs))

        return x

    def loss(self, predictions, targets):
        from torch.autograd import Variable
        lz = self.criterion(predictions[0], Variable(targets[0], requires_grad=False).cuda())
        return lz

    @staticmethod
    def get_name_model():
        return 'model_best.pth.tar'
