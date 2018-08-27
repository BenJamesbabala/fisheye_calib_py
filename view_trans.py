#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2
import numpy as np 

def find_extreme_points(img, size):
    '''
    函数说明：寻找棋盘格端点坐标

    Parameters:
        img - 经过内参数矫正后的图片
        size - 棋盘格大小(11, 7)
    Returns:
        extreme_points_list - 端点列表
    Modify:
        2018-08-27
    '''
    extreme_points_list = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, size, None)
    point_0 = [corners[0][0][0], corners[0][0][1]]
    point_1 = [corners[size[0]-1][0][0], corners[size[0]-1][0][1]]
    point_2 = [corners[-1][0][0], corners[-1][0][1]]
    point_3 = [corners[-size[0]][0][0], corners[-size[0]][0][1]]

    extreme_points_list.append(point_0)
    extreme_points_list.append(point_1)
    extreme_points_list.append(point_2)
    extreme_points_list.append(point_3)

    return extreme_points_list


def img_view_trans(img, points_src, points_dst):
    '''
    函数说明：视角转换为俯视图

    Parameters:
        img - 经过内参数矫正后的图像
        points_src - 原图像端点坐标列表
        points_dst - 目的图像端点坐标列表      
    Returns:
        vertical_view - 输出俯视图像
    Modify:
        2018-08-27
    '''
    perspective_mat = cv2.getPerspectiveTransform(np.array(points_src), np.array(points_dst))
    vertical_view = cv2.warpPerspective(img, perspective_mat, (img.shape[1], img.shape[0]))

    return vertical_view  


def make_points_dst(scale, size, center):
    '''
    函数说明：制作目标图像端点列表

    Parameters:
        scale - 缩放比例
        size - 棋盘格大小
        center - 中心点坐标
    Returns:
        points_dst - 目标图像端点列表
    Modify:
        2018-08-27
    '''
    points_dst = []
    w = (size[0]-1)*100*scale
    h = (size[1]-1)*100*scale
    point_0 = [center[0]-w/2, center[1]-h/2]
    point_1 = [center[0]+w/2, center[1]-h/2]
    point_2 = [center[0]+w/2, center[1]+h/2]
    point_3 = [center[0]-w/2, center[1]+h/2]

    points_dst.append(point_0)
    points_dst.append(point_1)
    points_dst.append(point_2)
    points_dst.append(point_3)

    return points_dst


if __name__ == '__main__':
    img = cv2.imread('./undistort_ground/undist_front.png')
    size = (11, 7)
    extreme_points_list = find_extreme_points(img, size)
    points_src = np.array(extreme_points_list)
    print(points_src)

    points_dst = make_points_dst(0.2, size, [540, 420])

    points_dst = np.float32(points_dst)
    print(points_dst)
    
    vertical_view = img_view_trans(img, points_src, points_dst)

    # color = (0, 0, 255)
    # for i in range(len(extreme_points_list)):
    #     cv2.circle(img, extreme_points_list[i], 5, color, -1)
    cv2.imshow('vertical_view', vertical_view)
    cv2.imwrite('vertical_front_0.8.png', vertical_view)
    cv2.waitKey()
