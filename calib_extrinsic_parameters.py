#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import cv2
import numpy as np 
import glob
from calibration_test import fisheye_calib

def calc_extrinsic(img, size):
    '''
    函数说明：相机外参数计算

    Parameters:
        img - 输入图像
        size - 角点行列数
    Returns:
        rvecs - 旋转向量
        tvecs - 平移向量
    Modify:
        2018-08-24
    '''
    # 设置寻找亚像素角点的参数
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    
    w = size[0]
    h = size[1]
    # print(w)
    # print(h)
        
    objp = np.zeros((1, w*h, 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        
    obj_points = []    # 存储3D点
    img_points = []    # 存储2D点

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    img_size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, size, None)
    # print(len(corners))
            
    if ret:
        obj_points.append(objp)
            
        # 在原角点的基础上寻找亚像素角点
        corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), subpix_criteria)  
        if corners2 is None:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (11,9), corners, ret)
        cv2.imshow('img', img)
        # cv2.waitKey(1000)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(obj_points))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(obj_points))]
    # print(len(obj_points))

    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

    rms, _, _, rvecs, tvecs = cv2.fisheye.calibrate(obj_points, img_points, img_size, K, D, rvecs, tvecs, 0, criteria)

    return rvecs, tvecs


if __name__ == '__main__':
    # camera = fisheye_calib((11, 7))
    # images = glob.glob('./ground/back.jpg')
    # K, D = camera.calibration(images)


    # img = cv2.imread('./ground/back.jpg')
    img = cv2.imread('./undistort_ground/left.png')
    cv2.imshow('undistort', img)
    rvecs, tvecs = calc_extrinsic(img, (11, 7))
    print(rvecs)
    print(tvecs)
    cv2.waitKey()
