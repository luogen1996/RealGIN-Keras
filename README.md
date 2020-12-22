# Global Inference Network- pytorch- Keras

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/luogen1996/MCN/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/keras-%237732a8)

This is the keras  implementation of the paper: A Real-time Global Inference Network for One-stage
Referring Expression Comprehension.  This repo provides codes for reproducing the results on refcoco-series datasets.

## Citation

    @article{zhou2019a,
    title={A Real-time Global Inference Network for One-stage Referring Expression Comprehension.},
    author={Zhou, Yiyi and Ji, Rongrong and Luo, Gen and Sun, Xiaoshuai and Su, Jinsong and Ding, Xinghao and Lin, Chiawen and Tian, Qi},
    journal={arXiv: Computer Vision and Pattern Recognition},
    year={2019}}

## Prerequisites

- Python 3.6

- tensorflow-1.9.0 for cuda 9 or tensorflow-1.14.0 for cuda10

- keras-2.2.4

- spacy (you should download the glove embeddings by running `spacy download en_vectors_web_lg` )

- Others (progressbar2, opencv, etc. see [requirement.txt](https://github.com/luogen1996/MCN/blob/master/requirement.txt))

## Data preparation

-  Follow the instructions of  [DATA_PRE_README.md](https://github.com/luogen1996/MCN/blob/master/data/README.md) to generate training data and testing data of RefCOCO, RefCOCO+ and RefCOCOg.

-  Download the pretrained weights of backbone (vgg and darknet). We provide pretrained weights of keras  version for this repo and another  darknet version for  facilitating  the researches based on pytorch or other frameworks.  All pretrained backbones are trained  on COCO 2014 *train+val*  set while removing the images appeared in the *val+test* sets of RefCOCO, RefCOCO+ and RefCOCOg (nearly 6500 images).  Please follow the instructions of  [DATA_PRE_README.md](https://github.com/luogen1996/MCN/blob/master/data/README.md) to download them.

## Training 

1. Preparing your settings. To train a model, you should  modify ``./config/config.json``  to adjust the settings  you want. 
2. Training the model. run ` train.py`  under the main folder to start training:
```
python train.py
```
3. Testing the model.  You should modify  the setting json to check the model path ``evaluate_model`` and dataset ``evaluate_set`` using for evaluation.  Then, you can run ` test.py`  by
```
python test.py
```
​	After finishing the evaluation,  a result file will be generated  in ``./result`` folder.

4. Training log.  Logs are stored in ``./log`` directory, which records the detailed training curve and accuracy per epoch. If you want to log the visualizations, please  set  ``log_images`` to ``1`` in ``config.json``.   By using tensorboard you can see the training details like below：
  <p align="center">
  <img src="https://github.com/luogen1996/MCN/blob/master/fig2.png" width="90%"/>
  </p>
  
## Acknowledgement

 Thanks for a lot of codes from [keras-yolo3](https://github.com/qqwweee/keras-yolo3) , [keras-retinanet](https://github.com/fizyr/keras-retinanet)  and the framework of  [darknet](https://github.com/AlexeyAB/darknet) using for backbone pretraining.

