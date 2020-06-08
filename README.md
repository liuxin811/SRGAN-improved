# SRGAN-improved

This program is an improved version based on srgan, the based code is in [srgan](https://github.com/tensorlayer/srgan) by [zsdonghao](https://github.com/zsdonghao)

### Prepare Data and Pre-trained VGG
1. You should download the the pretrained VGG16 model in [here](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) and put it in folder below in `config.py`<br>
```
path_vgg16="your_vgg16_folder\\"
``` 
2. You can download the dataset in [DIV2K](http://www.vision.ee.ethz.ch/ntire17/),and put the dataset in folders below in `config.py`<br> 
```
path_train_HR_orin = 'your_dir\\DIV2K_train_HR\\'
path_train_LR_orin = 'your_dir\\DIV2K_train_LR_bicubic\\X4\\'
path_valid_HR_orin = 'your_dir\\DIV2K_valid_HR\\'
path_valid_LR_orin = 'your_dir\\DIV2K_valid_LR_bicubic\\X4\\'
```
3. You should change the parameters suitalbe for your GPU/CPU in `config.py` <br>

parameters  | note 
  ------------- | ------------- 
 batch_size_init  | 初始化生成器使用的batchsize 
 n_epoch_init  | 初始化生成器使用的epoch数目 
 train_step_init  | （初始化生成器）使用所有剪裁好的图片中的1/train_step_init 作为训练集，train_step_init=1表示使用全部图片 
 lr_init  | 初始化生成器使用的学习率 
 batch_size_adv  | 对抗训练使用的batchsize 
 n_epoch_adv  | 对抗训练使用的epoch数目 
 train_step_adv  | （对抗训练）使用所有剪裁好的图片中的1/train_step_init 作为训练集，train_step_init=1表示使用全部图片  
 lr_adv  | 对抗训练使用的学习率 

4. To cut images in the dataset to the right size for training you should run the following code at least once.<br> 
```
config.py
```

### Dependecies
* tensorflow 
* tensorlayer
* numpy

### Run
* Start training
```
train.py
```
* Start Evaluating
```
evaluate.py
```
### citation
If you find this project useful, we would be grateful if you cite the TensorLayer paper：
```
@article{tensorlayer2017,
author = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
journal = {ACM Multimedia},
title = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
url = {http://tensorlayer.org},
year = {2017}
}
```

