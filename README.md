# SRGAN-improved

This program is an improved version based on srgan, the based code is in [srgan by zsdonghao](https://github.com/tensorlayer/srgan)

### Prepare Data and Pre-trained VGG
You should download the the pretrained VGG16 model in [here](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) and put it in folder below in `config.py`<br>
```
path_vgg16="your_vgg16_folder\\"
``` 
You can download the dataset in [DIV2K](http://www.vision.ee.ethz.ch/ntire17/),and put the dataset in folders below in `config.py`<br> 
```
path_train_HR_orin = 'your_dir\\DIV2K_train_HR\\'
path_train_LR_orin = 'your_dir\\DIV2K_train_LR_bicubic\\X4\\'
path_valid_HR_orin = 'your_dir\\DIV2K_valid_HR\\'
path_valid_LR_orin = 'your_dir\\DIV2K_valid_LR_bicubic\\X4\\'
```
You should initiate the parameters for training in `config.py`
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

To cut images in the dataset to the right size for the firsr time you should run.<br> 
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


