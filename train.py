# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:14:42 2020

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
import model
import config


def train_L1():
    '''使用L1 Loss训练生成器
    step:每隔多少张图片取一张加入训练集中，step=1表示使用全部图片
    start：从第几张图片开始
    '''
    train_loss=[]
    val_loss=[]
    G = model.get_G((config.batch_size_init, 56, 56, 3))
    #载入权重G
    if os.path.exists(os.path.join(config.path_model, 'g_init.h5')):
        G.load_weights(os.path.join(config.path_model, 'g_init.h5'))
    #训练数据集    
    train_ds, train_len = config.get_train_data(config.batch_size_init, step = config.train_step_init, start = 0)
    #验证数据集
    val_ds = config.get_valid_data(batch_size = config.batch_size_init, valid_size = config.batch_size_init, step = 10)
    print('训练集一共有{}张图片'.format(train_len)) 
    g_optimizer_init = tf.optimizers.Adam(learning_rate=config.lr_init)
    for epoch in range(config.n_epoch_init):
        step_time = time.time()
        epoch_loss=0
        i = 0 # 计数器
        G.train()
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                l1_loss = tl.cost.absolute_difference_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(l1_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            
            epoch_loss += l1_loss
            i += 1
        G.eval()
        v_loss = 0
        j=0
        for _, (lr_patchs, hr_patchs) in enumerate(val_ds):
            val_lr = lr_patchs
            val_hr = hr_patchs
            val_fr = G(val_lr)
            loss = tl.cost.absolute_difference_error(val_fr, val_hr, is_mean=True)
            v_loss += loss
            j+=1
        
        train_loss.append(epoch_loss/i)
        val_loss.append(v_loss/j)
        print('Epoch: [{}/{}] time: {:.3f}s : mean train loss of  is {:.5f}, mean valid loss is {:.5f}'.
              format(epoch, config.n_epoch_init,time.time() - step_time, epoch_loss/i,v_loss/j))
        train_loss_file = os.path.join(config.path_loss, 'train_loss_L1.txt') 
        np.savetxt(train_loss_file,train_loss)   
        valid_loss_file = os.path.join(config.path_loss, 'valid_loss_L1.txt')   
        np.savetxt(valid_loss_file,val_loss)
        G.save_weights(os.path.join(config.path_model, 'g_init_epoch_{}.h5'.format(epoch)))   
    G.save_weights(os.path.join(config.path_model, 'g_init.h5')) 
             
          
def train_adv():
     #with tf.device('/cpu:0'):
    
        
    '''initialize model'''
    G = model.get_G((config.batch_size_adv, 56, 56, 3))
    D = model.get_D((config.batch_size_adv, 224, 224, 3))
    vgg22 = model.VGG16_22((config.batch_size_adv, 224, 224, 3))
    
    G.load_weights(os.path.join(config.path_model, 'g_init.h5'))
    '''optimizer'''
    #g_optimizer_init = tf.optimizers.Adam(learning_rate=0.001)
    g_optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    d_optimizer = tf.optimizers.Adam(learning_rate=0.0001)

    G.train()
    D.train()
    vgg22.train()
    train_ds, train_len = config.get_train_data(config.batch_size_adv, step = config.train_step_adv, start = 0)
    print('训练集一共有{}张图片'.format(train_len))
    '''initialize generator with L1 loss in pixel spase'''
    
    '''train with GAN and vgg16-22 loss'''
    n_step_epoch = round(train_len // config.batch_size_adv)
    for epoch in range(config.n_epoch_adv):
        #一个epoch累计损失，初始化为0
        mse_ls=0; vgg_ls=0; gan_ls=0; d_ls=0
        #计数
        i=0
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                  
                fake_patchs = G(lr_patchs)
                feature22_fake = vgg22(fake_patchs) # the pre-trained VGG uses the input range of [0, 1]
                feature22_real = vgg22(hr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                
                #g_vgg_loss = 2e-3 * tl.cost.mean_squared_error(feature22_fake, feature22_real, is_mean=True)
                #g_gan_loss = -tf.reduce_mean(logits_fake)  
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                
                g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 1e-4 * tl.cost.mean_squared_error(feature22_fake, feature22_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
                
                mse_ls+=mse_loss
                vgg_ls+=vgg_loss
                gan_ls+=g_gan_loss
                d_ls+=d_loss
                i+=1
                
                ''' WGAN-gp 未完成
                d_loss = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
                g_loss = g_vgg_loss + g_gan_loss 
                eps = tf.random.uniform([batch_size, 1, 1, 1], minval=0., maxval=1.)
                interpolates  = eps*hr_patchs + (1. - eps)*fake_patchs
                grad = tape.gradient(D(interpolates), interpolates)
                slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1,2,3]))
                gradient_penalty = 0.1*tf.reduce_mean((slopes-1.)**2)
                
                d_loss += gradient_penalty
                '''
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            del(tape)
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.5f}, vgg:{:.5f}, adv:{:.5f}), d_loss: {:.5f}".format(
                        epoch, config.n_epoch_adv, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))
        print('~~~~~~~~~~~~Epoch {}平均损失~~~~~~~~~~~~~~~~'.format(epoch))
        print("Epoch: [{}/{}] time: {:.3f}s, g_loss(mse:{:.5f}, vgg:{:.5f}, adv:{:.5f}), d_loss: {:.5f}".format(
                        epoch, config.n_epoch_adv, time.time() - step_time, mse_ls/i, vgg_ls/i, gan_ls/i, d_ls/i))
        G.save_weights(os.path.join(config.path_model, 'g_adv.h5')) 
        print('\n')
    
if __name__ == '__main__':
    '''使用L1 Loss初始化生成器'''
    #with tf.device('/device:GPU:0'):
    #train_L1()
        
    '''Adversarial train 生成器损失(mse+vgg+gan)，使用CPU训练（我的GPU太小）'''
    with tf.device('/cpu:0'):
        train_adv()
        
             