# OCR
视频的关键帧抓取，文字图像的矫正，图像拼接。

decorate.py：函数测试所用装饰器。
image_stitching.py: 基于sift特征的图像拼接，此处正方向定义为逆时针旋转90度。
keyframe_exaction.py：基于帧差的视频关键帧提取。
rectify: 基于canny特征, 霍夫变换的文字图像矫正，此处设定的正方向为逆时针旋转90度，方便后续处理。(由于算法特性, 难以处理反拍以及图中其他干扰物体，针对后者目前认为可以进一步做个关键区域定位)