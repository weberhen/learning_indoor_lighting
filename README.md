# Learning to Estimate Indoor Lighting from 3D Objects

[Project page](http://vision.gel.ulaval.ca/~jflalonde/projects/illumPredict/index.html)

-------

## Dependencies:

* pytorch (https://pytorch.org/)
* Install the following lib/toolbox (also `python setup.py install`):
    * skylibs (https://github.com/weberhen/skylibs.git)
    * pytorch_toolbox (https://github.com/weberhen/pytorch_toolbox)
    
## Installation
`python setup.py install`

## Downloading the models/dataset (coming soon)

* [Link](http://rachmaninoff.gel.ulaval.ca/static/3dv18_illpred/models.tar.gz) to the pre-trained illumination predictor models (183MB).
* [Link](http://rachmaninoff.gel.ulaval.ca/static/3dv18_illpred/objects_ldr.tar.gz) to the LDR datasets to train the illumination predictor. (3.5GB)

* place the models at `/learning_indoor_lighting/IlluminationPredictor/models`. Ex: 
`models/bun_zipper_glossy/model_best.pth.tar`
* place the datasets at `/learning_indoor_lighting/Datasets`. Ex: `Datasets/indoor_hdr/train` and `Datasets/objects_ldr/bun_zipper_glossy/train`

## Visualizing training/testing
Activate visdom in another terminal:

`python -m visdom.server`

Then go to the terminal: localhost:8097

## Testing the autoencoder
`cd learning_indoor_lighting/AutoEncoder && python test.py`

## Testing the illumination predictor
`cd learning_indoor_lighting/IlluminationPredictor && python test.py`

## Citation
```
@inproceedings{weber_3dv_18,
  author    = {Henrique Weber and
               Donald Pr{\'{e}}vost and
               Jean{-}Fran{\c{c}}ois Lalonde},
  title     = {Learning to Estimate Indoor Lighting from 3D Objects},
  booktitle = {International Conference on 3D Vision, Verona, Italy},
  pages     = {199--207},
  year      = {2018},
}
```
