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


def split_undist(img, json_name):
    '''
    函数说明：输入一张有畸变的图片，输出无畸变的图片列表

    Parameters:
        img - 输入图像
        json_name - 标定参数json文件名
    Returns:
        undist_img_list - 无畸变图像列表
    Modify:
        2018-08-24
    '''
    undist_img_list = []
    image_list = img_split(img)
    param_dict = read_calib_param(json_name)

    directions = ['front', 'back', 'left', 'right']
    for i in range(len(directions)):
        param = param_dict[directions[i]]

        K = np.array(param['K'])
        D = np.array(param['D'])

        undist_img = undistort_img(image_list[i], K, D)
        undist_img_list.append(undist_img)
        del undist_img
    return undist_img_list


if __name__ == '__main__':
    img_name = './src_img/back/b1.png'
    json_name = './calibration.json'
    img = cv2.imread(img_name)   
    undist_img_list = split_undist(img, json_name)
    directions = ['front', 'back', 'left', 'right']

    for i in range(len(undist_img_list)):
        cv2.imshow('udistort', undist_img_list[i])
        cv2.imwrite('./%s.png' % directions[i], undist_img_list[i])
        cv2.waitKey(1000)

    
