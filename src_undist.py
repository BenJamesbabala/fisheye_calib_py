#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np 
import cv2
import glob
import json

from undistort_test import undistort_img
from img_split_test import img_split

def read_calib_param(filename):
    '''
    函数说明：从calibration.json中读取标定数据

    Parameters:
        filename - 文件名
    Returns:
        param_dict - 标定参数字典
    Modify:
        2018-08-23
    '''
    f = open(filename, encoding='utf-8')
    param_dict = json.load(f)

    return param_dict


if __name__ == '__main__':
    param_dict = read_calib_param('./calibration.json')
    img_name = './src_img/back/b1.png'
    img = cv2.imread(img_name)
    
    image_list = img_split(img)
    undist_img_list = []

    directions = ['front', 'back', 'left', 'right']
    for i in range(len(directions)):
        param = param_dict[directions[i]]
        # print(param)
        K = np.array(param['K'])
        D = np.array(param['D'])
        # print(K)
        # print(D)
        undist_img = undistort_img(image_list[i], K, D)
        undist_img_list.append(undist_img)
        del undist_img
    for i in range(len(undist_img_list)):
        cv2.imshow('udistort', undist_img_list[i])
        cv2.imwrite('./%s.png' % directions[i], undist_img_list[i])
        cv2.waitKey(1000)

    
