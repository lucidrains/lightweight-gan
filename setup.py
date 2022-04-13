import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['lightweight_gan']
from version import __version__

setup(
  name = 'lightweight-gan',
  packages = find_packages(),
  entry_points={
    'console_scripts': [
      'lightweight_gan = lightweight_gan.cli:main',
    ],
  },
  version = __version__,
  license='MIT',
  description = 'Lightweight GAN',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/lightweight-gan',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'generative adversarial networks'
  ],
  install_requires=[
    'adabelief-pytorch',
    'einops>=0.3',
    'fire',
    'kornia>=0.5.4',
    'numpy',
    'pillow',
    'retry',
    'torch>=1.10',
    'torchvision',
    'tqdm',
    'aim'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)