import pathlib
import numpy as np
from random import sample
from learning_indoor_lighting.tools.transformations import TonemapHDR
from learning_indoor_lighting.tools.utils import load_hdr_multichannel

if __name__ == "__main__":

    # return mean and standard deviation of a given dataset.
    # the code can deal with hdr or png/jpeg images.

    dataset_paths = ['../Datasets/objects_ldr/dragon_vrip_glossy/']

    # tn = ToneMapper()
    tonemap_hdr = TonemapHDR(gamma=1, percentile=90, max_mapping=.8)

    for dataset_path in dataset_paths:
        print('calculating mean and std of dataset in ',dataset_path)

        all_r = all_g = all_b = []

        all_dataset_names = sorted(
            pathlib.Path(dataset_path+'train/').glob('*.{}'.format('exr')))

        size_dataset = len(all_dataset_names)
        print(size_dataset)
        # find 1000 random files inside the folder to get the mean and std
        for _, filename in zip(range(1000), sample(all_dataset_names, size_dataset)):
            # print(str(filename))

            # img_hdr = EnvironmentMap(str(filename), 'LatLong')

            img_hdr, _, _ = load_hdr_multichannel(str(filename))

            _, img = tonemap_hdr(img_hdr, clip=True)

            masked_r = img[:, :, 0].flatten()[img[:, :, 0].flatten() != 0]
            masked_g = img[:, :, 1].flatten()[img[:, :, 1].flatten() != 0]
            masked_b = img[:, :, 2].flatten()[img[:, :, 2].flatten() != 0]

            all_r = np.append(all_r, masked_r)
            all_g = np.append(all_g, masked_g)
            all_b = np.append(all_b, masked_b)

        mean_r = np.mean(all_r)
        mean_g = np.mean(all_g)
        mean_b = np.mean(all_b)

        max_r = np.max(all_r)
        max_g = np.max(all_g)
        max_b = np.max(all_b)

        min_r = np.min(all_r)
        min_g = np.min(all_g)
        min_b = np.min(all_b)

        std_r = np.std(all_r)
        std_g = np.std(all_g)
        std_b = np.std(all_b)

        print('Dataset MEAN R - G - B')
        print(mean_r)
        print(mean_g)
        print(mean_b)

        print('Dataset STD R - G - B')
        print(std_r)
        print(std_g)
        print(std_b)

        file = open('{}/dataset_mean_std.txt'.format(dataset_path), 'w')
        file.write('Dataset MEAN R - G - B\n')
        file.write('{} {} {}\n'.format(mean_r, mean_g, mean_b))

        file.write('Dataset STD R - G - B\n')
        file.write('{} {} {}\n'.format(std_r, std_g, std_b))

        file.close()
