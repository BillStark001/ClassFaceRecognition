# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 23:32:26 2018

@author: BillStark001
"""

import os
import cv2
import main

base=''
save_path='Saved_Pics/'
align_path='Aligned_Pics/'

d=os.listdir(save_path)
for dd in d:
    if dd.endswith('.py'):continue
    ddd=os.listdir(save_path+dd)
    i=0
    for dddd in ddd:
        try:
            rects,faces=main.separateFace(cv2.imread((save_path+dd+'/'+dddd)))
        except Exception:
            pass
        if faces==[]:continue
        for f in faces:
            print(align_path+dd+'/'+'{:0>4}'.format(i)+'.jpg')
            try:
                os.mkdir(align_path+dd)
            except OSError:
                pass
            cv2.imwrite(align_path+dd+'/'+'{:0>4}'.format(i)+'.jpg',f)
            i=i+1