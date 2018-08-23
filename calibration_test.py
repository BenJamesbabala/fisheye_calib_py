#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np 
import cv2
import glob
import json

class fisheye_calib(object):

    def __init__(self, size):
        '''
        函数说明：初始化

        Parameters:
            无
        Returns:
            无
        Modify:
            2018-08-23
        '''
        self.size = size # 角点排布


    def calibration(self, filename_list):
        '''
        函数说明：标定算法

        Parameters:
            filename_list - 图片名称列表
        Returns:
            K - 内参数矩阵
            D - 畸变系数
        Modify:
            2018-08-23
        '''
        # 设置寻找亚像素角点的参数
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        # 获取标定板角点的位置
        w = self.size[0]
        h = self.size[1]

        objp = np.zeros((1, w*h, 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        
        obj_points = []    # 存储3D点
        img_points = []    # 存储2D点

        for fname in filename_list:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, self.size, None)
            
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
                cv2.waitKey(1000)
        
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(obj_points))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(obj_points))]

        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        rms, _, _, _, _ = cv2.fisheye.calibrate(obj_points, img_points, img_size, K, D, rvecs, tvecs, calibration_flags, criteria)

        return K, D


if __name__ == "__main__":
    camera = fisheye_calib((11, 9))
    images = glob.glob(".\\calib_img\\left\\*.png")
    K, D = camera.calibration(images)

    with open('left_data.json', 'w') as f:
        json.dump({'K':K.tolist(), 'D':D.tolist()}, f)

