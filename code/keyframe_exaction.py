# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:46:07 2020

@author: zhaochen
"""

import cv2
from math import *
import numpy as np
from scipy.signal import argrelextrema
from decorate import logger

def information_entropy(img):
    """
    计算图像的信息熵
    
    Parameters
    ----------
    img : ndarray
        输入灰度值图像的像素矩阵

    Returns
    -------
    res : double
        图像信息熵
    """
    
    prob = np.zeros(256, )
    
    # 计算各灰度值下出现的概率
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ind = img[i][j]
            prob[ind] += 1
    prob = prob / (img.shape[0] * img.shape[1])
    
    # 计算信息熵
    res = 0
    for i in range(prob.shape[0]):
        if prob[i] != 0:
            res -= prob[i] * math.log2(prob[i])
    return res

def rel_change(a, b):
    return (b - a) / max(a, b)


class Frame:
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
    
    def __lt__(self, other):
        return self.id < other.id
    
    def __gt__(self, other):
        return other.__lt__(self)
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __ne__(self, other):
        return not self.__eq__(other)


def smooth(data, window_len = 13,  window = 'hanning'):
    
    s = np.r_[2 * data[0] - data[window_len:1:-1], data, 2 * data[-1] - data[-1:-window_len:-1]]
    
    if window == 'flat':
        win = np.ones(window_len, 'd')
    elif window == 'hanning':
        win = getattr(np, window)(window_len)
    
    y = np.convolve(win / win.sum(), s, mode = 'same')
    return y[window_len - 1 : -window_len +1]
    

def diff_exaction(video_path, use_thresh=True, thresh=0.6, use_local_maximal=True, len_window=50):
    """
    计算相邻两帧间的差分图，根据
    Parameters
    ----------
    video_path : str
        视频地址.
    use_thresh : boolean, optional
        阈值筛选. The default is True.
    thresh : float, optional
        阈值. The default is 0.6.
    use_local_maximal : boolean, optional
        局部极值点筛选. The default is True.
    len_window : integer, optional
        窗口的宽度. The default is 50.

    Returns
    -------
    keyframe_id_set : TYPE
        经过筛选后的帧序列

    """
    cap = cv2.VideoCapture(video_path)
    
    ind = 0
    curr_frame, prev_frame = None, None  
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    while(success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        
        if curr_frame is not None and prev_frame is not None:
            diff = cv2.absdiff(curr_frame, prev_frame) # 获取差分图
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum  / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(ind, diff_sum_mean)
            frames.append(frame)
            
        prev_frame = curr_frame
        ind += 1
        success, frame = cap.read()
    
    cap.release()
    
    keyframe_id_set = set()
    
    # 根据阈值筛选
    if use_thresh:
        for i in range(1, len(frames)):
            if rel_change(np.float(frames[i-1].diff), np.float(frames[i].diff)) >= thresh:
                keyframe_id_set.add(frames[i].id)
    
    # 局部极值筛选
    if use_local_maximal:
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        
        for i in frame_indexes:
            keyframe_id_set.add(frames[i-1].id)
    
    return keyframe_id_set
    
@logger()
def exact(video_path, name = 'keyframe'):
    """
    按序列号抽取视频帧
    """
    
    keyframe_id_set = diff_exaction(video_path)
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    idx = 0
    while(success):
        if idx in keyframe_id_set:
            save_name = name + "_" + str(idx) + ".jpg"
            cv2.imwrite("../demo/keyframe/" + save_name, frame)
            keyframe_id_set.remove(idx)
        
        idx = idx + 1
        success, frame = cap.read()
    cap.release()
    
if __name__ == '__main__':
    exact("../demo/video/test.mp4")
    
    
    




