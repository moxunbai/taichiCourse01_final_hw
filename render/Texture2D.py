import taichi as ti
import cv2
import numpy as np

@ti.data_oriented
class Texture2D:

    def __init__(self,fn,reshape=True):
        _flag=cv2.IMREAD_UNCHANGED
        if fn.endswith('.hdr'):
            _flag = cv2.IMREAD_ANYDEPTH
        img = cv2.imread(fn, flags=_flag)
        self.width=img.shape[1]
        self.height=img.shape[0]

        self.data =img.reshape(self.width*self.height,img.shape[2]) if reshape else img

    def write_field(self):
        pass