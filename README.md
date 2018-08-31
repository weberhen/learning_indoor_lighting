# Learning to Estimate Indoor Lighting from 3D Objects

In this work, we propose a step towards a more accurate
prediction of the environment light given a single picture
of a known object. To achieve this, we developed a deep
learning method that is able to encode the latent space of
indoor lighting using few parameters and that is trained on
a database of environment maps. This latent space is then
used to generate predictions of the light that are both more
realistic and accurate than previous methods. To achieve
this, our first contribution is a deep autoencoder which is
capable of learning the feature space that compactly models
lighting. Our second contribution is a convolutional neural
network that predicts the light from a single image of an
object with known geometry and reflectance. To train these
networks, our third contribution is a novel dataset that contains
21,000 HDR indoor environment maps. The results
indicate that the predictor can generate plausible lighting
estimations even from diffuse objects.

-------

## Dependencies:

* OpenEXR
* Imath
* numpy
* pytorch (torch 0.4)
* visdom
* skylibs (https://github.com/soravux/skylibs)
* tqdm
* tensorflow (for tensorboard)
* pyyaml
