# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:41:34 2020

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



def evaluate():
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.path_valid_HR_orin, regx='.*.png', printable=False))[:]
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.path_valid_LR_orin, regx='.*.png', printable=False))[:]

    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.path_valid_LR_orin, n_threads=8)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.path_valid_HR_orin, n_threads=8)
    
    imid = 0  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    #print(valid_lr_img.shape)
    valid_hr_img = valid_hr_imgs[imid]
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    W, H = valid_hr_img.shape[0], valid_hr_img.shape[1]

    G = model.get_G([1, None, None, 3])
    G.load_weights(os.path.join(config.path_model, 'g_gan.h5'))
    G.eval()
    #网络输出图像
    gen_img = G(valid_lr_img).numpy()

    #插值放大的图像
    out_bicu = config.resize_img(valid_lr_img, (W, H))
    
    tl.vis.save_image(gen_img[0], os.path.join(config.path_pic, 'fh.png'))
    tl.vis.save_image(valid_lr_img[0], os.path.join(config.path_pic, 'rl.png'))
    tl.vis.save_image(valid_hr_img, os.path.join(config.path_pic, 'hr.png')) 
    tl.vis.save_image(out_bicu[0], os.path.join(config.path_pic, 'bh.png'))
    
    print('验证图像已保存在{}文件夹中'.format(config.path_pic))
    
    
if __name__ == '__main__':
    #with tf.device('/cpu'):
    evaluate()

