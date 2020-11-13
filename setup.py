from setuptools import setup, find_packages

setup(
  name = 'lightweight-gan',
  packages = find_packages(),
  version = '0.0.6',
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
    'torch',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)