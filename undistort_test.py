#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2
import numpy as np 
import json

def read_json(filename):
    '''
    函数说明：读取json格式数据

    Parameters:
        filename - json文件名
    Returns:
        K - 内参数矩阵
        D - 畸变系数矩阵
    Modify:
        2018-08-23
    '''
    f = open(filename, encoding='utf-8')
    calib_data = json.load(f)
    K = np.array(calib_data['K'])
    D = np.array(calib_data['D'])

    return K, D
    

def undistort_img(img, K, D):
    '''
    函数说明：图像畸变矫正

    Parameters:
        img - 输入畸变图像
        K - 内参数矩阵
        D - 畸变系数矩阵
    Returns:
        undist_img - 无畸变图像
    Modify:
        2018-08-23
    '''
    Knew = K.copy()
    Knew[(0, 1), (0, 1)] = 1.0 * K[(0, 1), (0, 1)]
    
    undist_img = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)

    return undist_img


if __name__ == "__main__":
    filename = '.\\left_data.json'
    K, D = read_json(filename)

    print(K)
    print(D)

    img = cv2.imread('.\\test.png')
    undist_img = undistort_img(img, K, D)
    cv2.imshow('undist', undist_img)
    cv2.imwrite('undist_test.png', undist_img)
    cv2.waitKey(1000)
    
    
