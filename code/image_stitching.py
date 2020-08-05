# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:49:12 2020

@author: zhaochen
"""

import cv2
import numpy as np
from decorate import logger

def sift_features(img):
    sift = cv2.xfeatures2d.SIFT_create() # 创建sift特征检测器
    kp, des = sift.detectAndCompute(img, None) # 特征点提取与描述生成
    kp_img = cv2.drawKeypoints(img, kp, None)
    
    return kp_img, kp, des

def good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    # matches = sorted(matches, key = lambda x: x[0].distance / x[1].distance)
    good = []
    
    for i, j in matches:
        if i.distance < 0.75 * j.distance:
            good.append(i)
            
    return good

@logger()
def siftImage(img_1, img_2):
    kp_image_1, kp_1, des_1 = sift_features(img_1)
    kp_image_2, kp_2, des_2 = sift_features(img_2)
    good_kp = good_match(des_1, des_2)
    
    if len(good_kp) > 4:
        ptsA = np.float32([kp_1[m.queryIdx].pt for m in good_kp]).reshape(-1, 1, 2)
        ptsB = np.float32([kp_2[m.trainIdx].pt for m in good_kp]).reshape(-1, 1, 2)
        
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
        imgOutput = cv2.warpPerspective(img_1, H, (img_1.shape[1]+img_2.shape[1], img_1.shape[0]))
        imgOutput[0:img_2.shape[0], 0:img_2.shape[1]] = img_2
    
    return imgOutput


if __name__ == '__main__':
    img_1 = cv2.imread('../demo/image_stitching/keyframe_85.jpg')
    img_2 = cv2.imread('../demo/image_stitching/keyframe_186.jpg') 
    imgOutput = siftImage(img_2, img_1)
    
    cv2.imwrite('../demo/image_stitching/result.jpg', imgOutput )
    
    