#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import cv2
import glob
import numpy as np 

def img_split(img):
    '''
    函数说明：分割图片

    Parameters:
        img - 输入图片
    Returns:
        image_list - 分割后图片列表(head, rear, left, right)
    Author:
        yangxu
    Modify:
        2018-08-23
    '''
    image_list = []
    w = int(img.shape[0]/2)
    h = int(img.shape[1]/2)

    image_head = img[0:w, 0:h]
    image_rear = img[w:2*w, 0:h]
    image_left = img[w:2*w, h:2*h]
    image_right = img[0:w, h:2*h]

    image_list.append(image_head)
    image_list.append(image_rear)
    image_list.append(image_left)
    image_list.append(image_right)

    return image_list


if __name__ == '__main__':
    images = glob.glob('./src_img/*/*.png')
    for i in range(len(images)):
        fname = images[i]
        img = cv2.imread(fname)
        image_list = img_split(img)

        cv2.imwrite('./split/front/%d.png' % i, image_list[0])
        cv2.imwrite('./split/back/%d.png' % i, image_list[1])
        cv2.imwrite('./split/left/%d.png' % i, image_list[2])
        cv2.imwrite('./split/right/%d.png' % i, image_list[3])
        del image_list[:]







