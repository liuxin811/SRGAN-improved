# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:34:39 2020

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
import config


#Generator
def get_G(input_shape):
    
    w_init = tf.random_normal_initializer(stddev=0.02)
    nin = Input(input_shape)
    n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init)(nin)
    '''上采样后直接加到网络末尾'''
    up_bicubic = tl.layers.UpSampling2d((4,4), method='bicubic')(nin)
    temp = n
    # B residual blocks
    for i in range(16):
        nn = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=None)(n)
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
        nn = Elementwise(tf.add)([n, nn])
        n = nn
    n = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = Elementwise(tf.add)([n, temp])
    # B residual blacks end

    n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)
    n = Conv2d(256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)
    nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init)(n)
    
    nn = Elementwise(tf.add)([nn, up_bicubic])
    G = Model(inputs=nin, outputs=nn)
    
    return G


def VGG16_22(input_shape):
    '''提取VGG16第二个池化层前的第二个卷积层激活前的特征'''
    
    '''Input layer'''
    net_in = Input(input_shape)
    
    """conv1"""
    conv1_1 = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu)(net_in)
    conv1_2 = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu)(conv1_1)
    pool1 = MaxPool2d((2, 2), (2, 2))(conv1_2)
    
    """conv2"""
    conv2_1 = Conv2d(128, (3, 3), (1, 1), act=tf.nn.relu)(pool1)
    conv2_2 = Conv2d(128, (3, 3), (1, 1), act=None)(conv2_1)#激活前的特征
    
    '''static model'''
    model = Model(inputs=net_in, outputs=conv2_2, name="vgg16_22")
    
    '''load weights'''
    params = []
    npz = np.load(os.path.join(config.path_vgg16, 'vgg16_weights.npz'))
    for val in sorted(npz.items()):
        params.append(val[1])   
    params = params[:8]
    tl.files.assign_weights(params, model)
    
    return model

'''Discriminator'''
def get_D(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    nin = Input(input_shape)
    n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(nin)

    n = Conv2d(df_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=gamma_init)(n)
    n = Elementwise(combine_fn=tf.add, act=lrelu)([n, nn])

    n = Flatten()(n)
    no = Dense(n_units=1, W_init=w_init)(n)
    D = Model(inputs=nin, outputs=no, name="discriminator")
    return D