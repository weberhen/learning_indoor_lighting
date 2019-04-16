# Learning to Estimate Indoor Lighting from 3D Objects

[Project page](http://vision.gel.ulaval.ca/~jflalonde/projects/illumPredict/index.html)

-------

## Dependencies:


Run
`python setup.py install`

Then install the following lib/toolbox (also `python setup.py install`):
* pytorch (https://pytorch.org/)
* skylibs (https://github.com/weberhen/skylibs.git)
* pytorch_toolbox (https://github.com/weberhen/pytorch_toolbox)

## Downloading the models/dataset (coming soon)

* [Link]() to the pre-trained illumination predictor models.
* [Link]() to the pre-trained autoencoder model (latent vector size=128).
* [Link]() to the LDR datasets to train the illumination predictor.

* place the models at `/learning_indoor_lighting/IlluminationPredictor/models`. Ex: 
`models/indoor_hdr` and `models/objects_ldr`
* place the datasets at `/learning_indoor_lighting/Datasets`. Ex: `Datasets/bun_zipper_scene_glossy`

## Visualizing training/testing
Activate visdom in another terminal:

`python -m visdom.server`

Then go to the terminal: localhost:8097

## Testing the autoencoder
`cd learning_indoor_lighting/AutoEncoder && python test.py`

## Testing the illumination predictor
`cd learning_indoor_lighting/IlluminationPredictor && python test.py`

