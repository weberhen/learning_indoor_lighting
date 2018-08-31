# Henrique Weber, 2018
# Data transform functions


import numpy as np
import torch


class RandomRotation(object):
    """
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """

    @staticmethod
    def sph2cart(phi, theta, r):
        import math
        points = np.zeros(3)
        points[0] = r * math.sin(phi) * math.cos(theta)
        points[1] = r * math.sin(phi) * math.sin(theta)
        points[2] = r * math.cos(phi)
        return points

    @staticmethod
    def set_rotation(x, y, z):
        from illuminnet import angles as ea
        matrix = np.eye(3, dtype=np.float32)
        matrix[0:3, 0:3] = ea.euler2mat(x, y, z)

        return matrix

    def random_direction(self):
        import math
        # Random pose on a sphere : https://www.jasondavies.com/maps/random-points/
        theta = np.random.uniform(0, 1) * math.pi * 2
        phi = math.acos((2 * (np.random.uniform(0, 1))) - 1)
        x, y, z = self.sph2cart(phi, theta, 1)
        return self.set_rotation(x, y, z)

    def __call__(self, sample):
        from envmap import EnvironmentMap
        image = EnvironmentMap(64, 'LatLong')
        image.data = sample
        rotation = self.random_direction()
        img_hdr = image.rotate('DCM', rotation).data.astype('float32')
        sample = img_hdr
        return sample


class Log1p(object):
    """
        Apply log(input+1)
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """

    def __init__(self, undo_trans=False):
        self.undo_trans = undo_trans  # if True, performs the oposite operation (to remove the transformation effect
        # from the network prediction

    def __call__(self, sample):
        if self.undo_trans:  # if True, performs the oposite operation (to remove the transformation effect
            # from the network prediction
            return self.undo(sample)
        else:
            img = np.log1p(sample).copy()
            sample = img
            return sample

    def undo(self, sample):
        img = np.expm1(sample).copy()
        assert not np.isnan(img).any()
        sample = img
        return sample


class HorizontalFlip(object):
    """
        Randomly flip an image.
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """

    def __call__(self, sample):
        random_flip = np.random.randint(0, 1)
        if random_flip:
            img = np.flip(sample, axis=1).copy()
        else:
            img = sample
        sample = img
        return sample


class NumpyImage2Tensor(object):
    def __init__(self, undo_trans=False):
        self.undo_trans = undo_trans  # if True, performs the oposite operation (to remove the transformation effect
        # from the network prediction

    # from the network prediction
    def __call__(self, sample):
        if self.undo_trans:
            return self.undo(sample)
        else:
            import torch
            # numpy image: H x W x C
            # torch image: C X H X W
            img = sample
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)
            sample = img
            return sample

    def undo(self, sample):
        img = sample
        img = img.data.cpu().numpy().squeeze().transpose(1, 2, 0)
        sample = img
        return sample


class Scale(object):

    def __init__(self, random_perturbation, undo_trans=False):
        """
        :param s: scalar
        usually we need to rescale the ldr images (specially the diffuse ones) by a factor of 120. So here we apply the
        same operation to the envmaps to let the autoencoder robust to this kind of scaling
        """
        self.undo_trans = undo_trans  # if True, performs the oposite operation (to remove the transformation effect
        # from the network prediction
        self.s = [120, 120, 120]
        self.random_perturbation = random_perturbation
        self.perturbation_range = 2

    def __call__(self, sample, scales=np.array([])):
        if self.undo_trans:
            return self.undo(sample)
        else:
            local = sample.copy()
            if scales.size:
                self.s = scales
            else:
                self.s = [120, 120, 120]
            for i in range(3):
                if self.random_perturbation:
                    r = np.random.uniform(-self.perturbation_range / 2, self.perturbation_range / 2)
                    # randomly set the scale to twice larger or twice smaller, if perturbation range=2 for ex.
                    s_perturbed = self.s[i] * np.power(2, r)
                    local[:, :, i] = s_perturbed * local[:, :, i]
                else:
                    local[:, :, i] = self.s[i] * local[:, :, i]
            sample = local
            return sample

    def undo(self, sample):
        img = sample.squeeze().copy()
        for i in range(3):
            img[:, :, i] = img[:, :, i] / self.s[i]
        sample = img
        return sample


class Normalize(object):
    """
        will normalize each channel of the torch.*Tensor, i.e.
        channel = (channel - mean) / std

        input : torch tensor batch of images : [N, C, H, W]
        output : torch tensor batch of images : [N, C, H, W]
    """

    def __init__(self, mean, std, ignore_zero=False, undo_trans=False):
        """

        :param mean: iterator with value per channel : ex : [R, G, B]
        :param std: iterator with value per channel : ex : [R, G, B]
        """
        self.undo_trans = undo_trans  # if True, performs the oposite operation (to remove the transformation effect
        # from the network prediction
        self.ignore_zero = ignore_zero
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        local = sample.clone()
        if self.undo_trans:
            return self.undo(local)
        else:
            if self.ignore_zero:
                mask = np.all(local.numpy()[0:3] > 0, axis=0)
                mask = torch.from_numpy(mask * 1).byte()
            for _, t, m, s in zip(range(3), local, self.mean, self.std):
                if self.ignore_zero:
                    t[mask].sub_(m).div_(s)
                else:
                    t.sub_(m).div_(s)
            return local

    def undo(self, sample):
        if self.ignore_zero:
            mask = np.all(sample.numpy()[0:3] > 0, axis=0)
            mask = torch.from_numpy(mask * 1).byte()
        local = sample.clone()
        for i, t, m, s in zip(range(3), sample, self.mean, self.std):
            if self.ignore_zero:
                local[i] = t[mask].mul(s).add(m)
            else:
                local[i] = t.mul(s).add(m)
        return local


class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma, percentile, max_mapping):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip):
        power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img = np.clip(tonemapped_img, 0, 1)

        return alpha, tonemapped_img.astype('float32')


class ToFloat(object):
    def __call__(self, numpy):
        return numpy.astype(np.float32)
