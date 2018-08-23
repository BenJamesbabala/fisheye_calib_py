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
        mtx - 内参数矩阵
        dist - 畸变系数矩阵
    Modify:
        2018-08-23
    '''
    f = open(filename, encoding='utf-8')
    calib_data = json.load(f)
    mtx = np.array(calib_data['mtx'])
    dist = np.array(calib_data['dist'])

    return mtx, dist
    

def undistort_img(img, mtx, dist):
    '''
    函数说明：图像畸变矫正

    Parameters:
        img - 输入畸变图像
        mtx - 内参数矩阵
        dist - 畸变系数矩阵
    Returns:
        undistort_img - 无畸变图像
    Modify:
        2018-08-23
    '''
    mtx_new = mtx.copy()
    mtx_new[(0, 1), (0, 1)] = 0.4 * mtx[(0, 1), (0, 1)]
    
    undist_img = cv2.fisheye.undistortImage(img, mtx, dist, mtx_new)

    return undist_img


if __name__ == "__main__":
    filename = '.\\left_data.json'
    mtx, dist = read_json(filename)

    # print(mtx)
    # print(dist)

    img = cv2.imread('.\\test.png')
    undist_img = undistort_img(img, mtx, dist)
    cv2.imshow('undist', undist_img)
    cv2.imwrite('undist_test.png', undist_img)
    cv2.waitKey(1000)
    
    
