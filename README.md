## 'Lightweight' GAN

[![PyPI version](https://badge.fury.io/py/lightweight-gan.svg)](https://badge.fury.io/py/lightweight-gan)

Implementation of an extremely <a href="https://openreview.net/forum?id=1Fqg133qRaI">'lightweight' GAN</a> proposed in ICLR 2021, in Pytorch. The main contributions of the paper is a skip-layer excitation in the generator, paired with autoencoding self-supervised learning in the discriminator. Quoting the one-line summary "converge on single gpu with few hours' training, on 1024 resolution sub-hundred images".

## Install

```bash
$ pip install lightweight-gan
```

## Usage

```bash
$ lightweight_gan --data ./path/to/images --image-size 512 --aug-prob 0.2
```

## Status

512x512 Oxford flowers in progress, results pending. Feel free to try yourself in the interim.

## Citations

```bibtex
@inproceedings{
    anonymous2021towards,
    title={Towards Faster and Stabilized {\{}GAN{\}} Training for High-fidelity Few-shot Image Synthesis},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=1Fqg133qRaI},
    note={under review}
}
```

```bibtex
@inproceedings{
    anonymous2021is,
    title={Is Attention Better Than Matrix Decomposition?},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=1FvkSpWosOl},
    note={under review}
}
```
