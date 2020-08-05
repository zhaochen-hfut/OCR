# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:03:09 2020

@author: zhaochen
"""

import cv2
from math import sin, cos, fabs, radians
import numpy as np
from decorate import logger

def rad2theta(theta):
    return theta / np.pi * 180


def rotateImage(img, degree):
    """
    根据角度旋转图像
    """
    
    
    height, width = img.shape[:2]
    
    # 计算旋转后的图像尺寸
    heightNew = int(width*fabs(sin(radians(degree))) + height*fabs(cos(radians(degree))))
    widthNew  = int(height*fabs(sin(radians(degree))) + width*fabs(cos(radians(degree)))) 
    
    # 旋转
    M = cv2.getRotationMatrix2D((width/2, height/2) , degree, 1)
    
    # 旋转中心点校准
    M[0,2] += (widthNew - width) / 2
    M[1,2] += (heightNew - height) / 2
     
    dst = cv2.warpAffine(img, M, (widthNew, heightNew), borderValue=(255,255,255))
    
    return dst
    
def CalcDegree(img, flag = True):
    """
    Canny算子滤波, 霍夫变换求直线, 统计所有直线取平均, 计算旋转角度
    
    Parameters
    ----------
    img : ndarray
        bgr像素矩阵.
    flag : boolean, optional
        是否在图中打印出直线. The default is True.

    Returns
    -------
    dst : TYPE
        旋转后的图像.
    angle : float
        旋转角度.
    """
    
    midImage = cv2.Canny(img, threshold1=50, threshold2=200, apertureSize=3) # canny算子
    dstImage = cv2.cvtColor(midImage, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(midImage, rho=1, theta=np.pi /180, threshold=320) #霍夫变换
    
    if lines is None:
        lines = cv2.HoughLines(midImage, rho=1, theta=np.pi /180, threshold=250)
    
    if lines is None:
        lines = lines = cv2.HoughLines(midImage, rho=1, theta=np.pi /180, threshold=150)
    
    total = 0 # 统计所有检测到的直线的角度数
    
    for line in lines:
        rho, theta = line[0][0], line[0][1]
        
        
        # 极坐标空间向笛卡尔坐标空间变换
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        
        pt1_x = np.round(x0 + 1000 * (-b)).astype(dtype = np.int32)
        pt1_y = np.round(y0 + 1000 * a).astype(dtype =  np.int32)
        pt2_x = np.round(x0 - 1000 * (-b)).astype(dtype =  np.int32)
        pt2_y = np.round(y0 - 1000 * a).astype(dtype =  np.int32)
        #print(theta)
        total += theta
        
        if flag:  
            cv2.line(dstImage, (pt1_x, pt1_y), (pt2_x, pt2_y), (55, 100, 195), 1, cv2.LINE_AA)
        
        cv2.imwrite('../pic/test1.jpg', dstImage)
    
    print(len(lines))
    average = total / len(lines)
    angle = rad2theta(average)
    
    dst = rotateImage(img, angle)
    
    return dst, angle
    
@logger()
def ImageRecify(img):
    dst, degree = CalcDegree(img, False)
    print('angle: ' + str(degree))
    
    
    dst = dst[0:dst.shape[0], 0:dst.shape[1]]
    
    cv2.imwrite('../demo/rectify/result.jpg', dst)
    
    
if __name__ == '__main__':
    img = cv2.imread('../demo/rectify/keyframe_12.jpg') 
    ImageRecify(img)
    