from setuptools import setup

setup(
    name='learning_indoor_lighting',
    packages=['learning_indoor_lighting',
              'learning_indoor_lighting.AutoEncoder',
              'learning_indoor_lighting.IlluminationPredictor',
              'learning_indoor_lighting.tools'],
    install_requires=['numpy', 'tqdm', 'visdom', 'pyyaml','openexr', 'torchvision', 'rotlib', 'imageio']
)