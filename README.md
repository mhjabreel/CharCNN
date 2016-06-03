# CharCNN

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains my implementation using TensorFlow for text classification from character-level using convolutional networks. It can be used to reproduce the results in the following article:

Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems 28 (NIPS 2015)

![Alt text](ccnn.png?raw=true "The model")

## How to use
First, specify the training and testing data sources in the config.py file.

Then, run the training.py file as below:
```sh
$ python training.py
```

License

Copyright (c) 2016 Minh Ngo

The source code is distributed under the MIT license.
