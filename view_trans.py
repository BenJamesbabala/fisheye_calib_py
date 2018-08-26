#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2
import numpy as np 
import json
import glob

from calib_extrinsic_parameters import calc_extrinsic

def img_view_change(img, rvecs):
    '''
    函数说明：将给定图像根据旋转矩阵进行视角转换

    Parameters:
        img - 输入图像
        rmat - 旋转矩阵
    Returns:
        vertical_img - 输出经过视角转换后的俯视图
    Modify:
        2018-08-26
    '''
    rmat = cv2.Rodrigues(rvecs)[0]
    return rmat


if __name__ == '__main__':
    img = cv2.imread('./undistort_ground/back.png')
    size = (11, 7)
    rvecs, tvecs = calc_extrinsic(img, size)
    rvecs_T = rvecs[0].tolist()
    rotate_list = [] 
    print(len(rvecs_T))
    # print(rvecs_T[1][0])
    for i in range(len(rvecs_T)):
        rotate_list.append(rvecs_T[i][0])
    print(rotate_list)
    om = np.array(rotate_list)
    rmat = cv2.Rodrigues(om)[0]
    print(rmat)
    vertical_img = cv2.warpPerspective(img, rmat, (img.shape[1], img.shape[0]))
    cv2.imshow('vertical', vertical_img)
    cv2.waitKey()
