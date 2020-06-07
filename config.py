# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:25:28 2020

@author: liuxin
"""

import numpy as np
import os 
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense, MaxPool2d
from tensorlayer.models import Model
import time
import cv2

#初始化生成器训练参数
batch_size_init = 6
n_epoch_init = 3
train_step_init = 100 #使用所有剪裁好的图片中的1/2作为训练集，train_step_init=1表示使用全部图片
lr_init = 1e-4

#对抗训练参数
batch_size_adv = 8
n_epoch_adv = 3
train_step_adv = 10 #使用所有剪裁好的图片中的1/10作为训练集，train_step_init=1表示使用全部图片
lr_adv = 1e-4

'''原图像所在的文件'''
path_train_HR_orin = 'D:\\SR\\DIV2K\\DIV2K_train_HR\\'
path_train_LR_orin = 'D:\\SR\\DIV2K\\DIV2K_train_LR_bicubic\\X4\\'
path_valid_HR_orin = 'D:\\SR\\DIV2K\\DIV2K_valid_HR\\'
path_valid_LR_orin = 'D:\\SR\\DIV2K\\DIV2K_valid_LR_bicubic\\X4\\'


'''剪裁后的图像所在的文件'''
path_train_HR_crop = 'D:\\SR\\DIV2K\\train_HR\\'
path_train_LR_crop = 'D:\\SR\\DIV2K\\train_LR\\'
path_valid_HR_crop = 'D:\\SR\\DIV2K\\valid_HR\\'
path_valid_LR_crop = 'D:\\SR\\DIV2K\\valid_LR\\'

'''模型权重保存的路径'''
path_model = 'D:\\SR\\model\\'
'''evaluate时生成图像保存的路径'''
path_pic = 'D:\\SR\\pic\\'
'''训练损失保存路径'''
path_loss = 'D:\\SR\\loss\\'
path_vgg16 = 'D:\\SR\\VGG16\\'

'''创建文件夹'''
def create_dir(path):
    if not(os.path.exists(path)):
        os.mkdir(path)
        return True
    else:
        return False
    

def prepare_data(sour_path, dest_path, side_len):
    '''将DIV2K中的大图切成小图，得到剪裁后数据集
    sour_path:原图像所在的文件夹
    dest_path:小图保存的目标文件
    side_len:小图的边长
    '''
    
    fileList = os.listdir(sour_path)
    if len(fileList)==0 :
        print('警告：{}中没有图片！'.format(sour_path))
        
    num=1 #计数
    '''文件夹必须不存在才操作，避免重复剪裁图像'''
    if create_dir(dest_path):
        for img_name in fileList:
            img = cv2.imread(sour_path + img_name)            
            #宽度能剪裁几张
            wn = int(img.shape[0]/side_len)
            #高度能剪裁几张
            hn = int(img.shape[1]/side_len)
            for i in range(wn):
                for j in range(hn):
                    crop_img = img[i*side_len : (i+1)*side_len, j*side_len : (j+1)*side_len]
                    cv2.imwrite(dest_path + str(num) + '.png', crop_img)
                    if num%1000==0:
                        print('{}已经生成{}张图片'.format(dest_path, num))
                    num += 1            
        print('{}已经生成{}张图片'.format(dest_path, num))
    else:
        fileList = os.listdir(dest_path)
        print('{}文件夹有{}张图片'.format(dest_path,len(fileList)))
                
            
'''调整图像大小'''
def resize_img(lr_patchs, size=(224,224)):
    return np.array([tl.prepro.imresize(x, size=size) for x in lr_patchs])
            

def get_train_data(batch_size, step = 1, start = 0):
    train_img_list = tl.files.load_file_list(path=path_train_HR_crop, regx='.*.png', printable=False)[start:-1:step]
    train_hr_imgs = tl.vis.read_images(train_img_list, path=path_train_HR_crop, n_threads=8, printable=False)
    train_lr_imgs = tl.vis.read_images(train_img_list, path=path_train_LR_crop, n_threads=8, printable=False)
    train_len = len(train_img_list)
    def generator_train():
        for img_lr, img_hr in zip(train_lr_imgs, train_hr_imgs):
            img_hr = img_hr / (255. / 2.) - 1.
            img_lr = img_lr / (255. / 2.) - 1.
            yield img_lr, img_hr
            
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32))
    train_ds = train_ds.shuffle(buffer_size = 128)
    train_ds = train_ds.prefetch(buffer_size = batch_size)
    train_ds = train_ds.batch(batch_size = batch_size, drop_remainder=True)
    return train_ds, train_len
    
def get_valid_data(batch_size, valid_size, step = 1):
    if valid_size < batch_size:
        print('警告：get_valid_data函数中valid_size不能小于batch_size')
        valid_size = batch_size
    img_list = tl.files.load_file_list(path=path_valid_HR_crop, regx='.*.png', printable=False)[::step]
    valid_size = valid_size if valid_size <= len(img_list) else len(img_list)
    img_list = img_list[:valid_size]
    val_hr_imgs = tl.vis.read_images(img_list, path = path_valid_HR_crop, n_threads=8, printable=False)
    val_lr_imgs = tl.vis.read_images(img_list, path = path_valid_LR_crop, n_threads=8, printable=False)
    
    def generator_train():
        for img_lr, img_hr in zip(val_lr_imgs, val_hr_imgs):
            img_hr = img_hr / (255. / 2.) - 1.
            img_lr = img_lr / (255. / 2.) - 1.
            yield img_lr, img_hr
            
    val_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32))
    val_ds = val_ds.batch(batch_size = batch_size, drop_remainder=True)
    return val_ds


if __name__ == '__main__':
    create_dir(path_model)
    create_dir(path_pic)
    create_dir(path_loss)

    prepare_data(path_train_HR_orin, path_train_HR_crop, 224)
    prepare_data(path_train_LR_orin, path_train_LR_crop, 56)
    prepare_data(path_valid_HR_orin, path_valid_HR_crop, 224)
    prepare_data(path_valid_LR_orin, path_valid_LR_crop, 56)


