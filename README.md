# TuSimple-DUC

by Panqu Wang, Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, Xiaodi Hou, and Garrison Cottrell.

## Introduction

This repository is for [Understanding Convolution for Semantic Segmentation](https://arxiv.org/abs/1702.08502) (WACV 2018), which achieved state-of-the-art result on the CityScapes, PASCAL VOC 2012, and Kitti Road benchmark.

## Requirement

We tested our code on:

Ubuntu 16.04, Python 2.7 with

[MXNet (0.11.0)](https://github.com/TuSimple/mxnet), numpy(1.13.1), cv2(3.2.0), PIL(4.2.1), and cython(0.25.2)

## Usage

1. Clone the repository:

   ```shell
   git clone git@github.com:TuSimple/TuSimple-DUC.git
   python setup.py develop --user
   ```

2. Download the pretrained model from [Google Drive](https://drive.google.com/open?id=0B72xLTlRb0SoREhISlhibFZTRmM).

3. Build MXNet (only tested on the TuSimple version):

   ```shell
   git clone --recursive git@github.com:TuSimple/mxnet.git
   vim make/config.mk (we should have USE_CUDA = 1, modify USE_CUDA_PATH, and have USE_CUDNN = 1 to enable GPU usage.)
   make -j
   cd python
   python setup.py develop --user
   ```

   For more MXNet tutorials, please refer to the [official documentation](https://mxnet.incubator.apache.org/install/index.html).

3. Training:

   ```shell
   cd train
   python train_model.py ../configs/train/train_cityscapes.cfg
   ```

   The paths/dirs in the ``.cfg`` file need to be specified by the user.

4. Testing

   ```
   cd test
   python predict_full_image.py ../configs/test/test_full_image.cfg
   ```

   The paths/dirs in the ``.cfg`` file need to be specified by the user.

5. Results:

   Modify the ``result_dir`` path in the config file to save the label map and visualizations. The expected scores are:

   (single scale testing denotes as 'ss' and multiple scale testing denotes as 'ms')

   - ResNet101-DUC-HDC on CityScapes testset (mIoU): 79.1(ss) / 80.1(ms)
   - ResNet152-DUC on VOC2012 (mIoU): 83.1(ss)

## Citation

If you find the repository is useful for your research, please consider citing:

    @article{wang2017understanding,
      title={Understanding convolution for semantic segmentation},
      author={Wang, Panqu and Chen, Pengfei and Yuan, Ye and Liu, Ding and Huang, Zehua and Hou, Xiaodi and Cottrell, Garrison},
      journal={arXiv preprint arXiv:1702.08502},
      year={2017}
    }

## Questions

Please contact panqu.wang@tusimple.ai or pengfei.chen@tusimple.ai .
