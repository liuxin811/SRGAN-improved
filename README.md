# SRGAN-improved

This program is an improved version based on srgan, the based code is in [srgan by zsdonghao](https://github.com/tensorlayer/srgan)

### Prepare Data and Pre-trained VGG
You should download the the pretrained VGG16 model in [here](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) and put it in `config.py`<br>
```
path_vgg16="your_vgg16_folder/"
``` 
You can download the dataset in [DIV2K](http://www.vision.ee.ethz.ch/ntire17/),and put the dataset in folders below<br> 
```
path_train_HR_orin = 'your_dir\\DIV2K_train_HR\\'
path_train_LR_orin = 'your_dir\\DIV2K_train_LR_bicubic\\X4\\'
path_valid_HR_orin = 'your_dir\\DIV2K_valid_HR\\'
path_valid_LR_orin = 'your_dir\\DIV2K_valid_LR_bicubic\\X4\\'
```
You should run `config.py` to cut images in the dataset to the right size for the firsr time.<br> 
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


