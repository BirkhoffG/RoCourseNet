# RoCourseNet: Distributionally Robust Training of a Prediction Aware Recourse Model

[![Arxiv](https://img.shields.io/badge/Arxiv-2206.00700-orange)](https://arxiv.org/pdf/2206.00700.pdf)

This repo contains code to reproduce our paper published at [CIKM 2023](https://arxiv.org/pdf/2206.00700.pdf).

To cite this paper:


```bibtex
@inproceedings{guo2021rocoursenet,
    author={Guo, Hangzhi and Jia, Feiran and Chen, Jinghui and Squicciarini, Anna and Yadav, Amulya},
    title = {RoCourseNet: Robust Training of a Prediction Aware Recourse Model},
    year = {2023},
    isbn = {9798400701030},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3583780.3615040},
    doi = {10.1145/3583780.3615040},
    booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
    numpages = {15},
    location = {Birmingham, United Kingdom},
    series = {CIKM ’23}
}

```

## Install

This project uses 
[jax-relax](https://github.com/BirkhoffG/ReLax/tree/master) (a fast and scalable recourse explanation library).
Ths library is highly scalable and extensible, which enables our experiments to be finished within 30 minutes.
In contrast, a pytorch implementation of RoCourseNet takes around 12 hours to run.

```sh
pip install -e ".[dev]" --upgrade
```

## Run Experiments

Running `scripts.experiment.py` with different arguments will reproduce results in our paper. For example,

1. Train and Evaluate RoCourseNet on Loan Application Dataset:

```sh
python -m scripts.experiment.py -d loan
```

2. Train and Evaluate CounterNet on Loan Application Dataset:

```sh
python -m scripts.experiment.py -m CounterNet -d loan
```

3. Train and Evaluate ROAR on Loan Application Dataset:

```sh
python -m scripts.experiment.py -m ROAR -d loan
```
