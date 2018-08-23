#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np 
import cv2
import glob
import json

class fisheye_calib(object):

    def __init__(self, max_iter, eps, size):
        '''
        函数说明：初始化

        Parameters:
            无
        Returns:
            无
        Modify:
            2018-08-23
        '''
        self.max_iter = max_iter # 最大循环次数
        self.eps = eps # 最大误差容限
        self.size = size # 角点排布


    def calibration(self, filename_list):
        '''
        函数说明：标定算法

        Parameters:
            filename_list - 图片名称列表
        Returns:
            ret - 
            mtx - 内参数矩阵
            dist - 畸变系数
            rvecs - 旋转向量
            tvecs - 平移向量
        Modify:
            2018-08-23
        '''
        # 设置寻找亚像素角点的参数
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, self.max_iter, self.eps)

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
                corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)  
                if corners2 is None:
                     img_points.append(corners2)
                else:
                    img_points.append(corners)
        
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(obj_points))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(obj_points))]

        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(obj_points, img_points, img_size, K, D, rvecs, tvecs, calibration_flags, criteria)

        return ret, mtx, dist, rvecs, tvecs


if __name__ == "__main__":
    camera = fisheye_calib(30, 0.001, (11, 9))
    images = glob.glob(".\\calib_img\\left\\*.png")
    ret, mtx, dist, rvecs, tvecs = camera.calibration(images)

    # print("mtx:\n", mtx)        # 内参数矩阵
    # print("dist:\n", dist)      # 畸变系数   

    with open('left_data.json', 'w') as f:
        json.dump({'mtx':mtx.tolist(), 'dist':dist.tolist()}, f)

