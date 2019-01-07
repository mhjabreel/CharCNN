# CharCNN

![MIT](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains my implementation using [Textify](https://github.com/mhjabreel/Textify) for Character-level Convolutional Networks for Text Classification. It can be used to reproduce the results in the following article:

Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems 28 (NIPS 2015)

![Alt text](ccnn.png?raw=true "The model")

## How to use
First, install Textify:.
```sh
    pip install git+https://github.com/mhjabreel/Textify.git
```

Then, run the follwoing command to train the model:
```sh
    textify train_and_eval --config configs/model.yml configs/data.yml configs/train.yml
```

## License

Copyright (c) 2016 Mohammed Jabreel

The source code is distributed under the MIT license.
