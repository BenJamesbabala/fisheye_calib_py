#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2
import numpy as np 
import glob
import json
from calibration_test import *

if __name__ == "__main__":
    camera = fisheye_calib((11, 9))
    directions = ['front', 'back', 'left', 'right']

    calib_param = {}
    for direction in directions:
        filename = './split/%s/*.png' % direction
        # print(filename)
        images = glob.glob(filename)
        K, D = camera.calibration(images)
        calib_param[direction] = {'K':K.tolist(), 'D':D.tolist()} 

    with open('./calibration.json', 'w') as f:
        json.dump(calib_param, f)