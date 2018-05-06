# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:32:47 2018
@author: BillStark001
"""

import dlib
from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
try:
    assert d
except:
    from mtcnn.mtcnn import MTCNN
    d = MTCNN()

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

#BASE_IMG=np.zeros((112,112,3))
'''
BASE_LANDMARKS=np.matrix([[15,16,19,21,25,31,39,47,56,65,73,80,87,
91,93,96,98,23,28,35,43,50,63,69,77,84,89,57,56,56,56,48,52,55,59,62,32,36,41,
45,41,36,66,70,75,80,76,71,39,45,50,55,59,65,71,65,59,55,50,44,41,50,54,59,69,59,55,50],
[47,58,69,79,89,97,103,109,110,108,103,96,89,
80,70,60,49,44,39,37,38,41,41,38,38,39,44,52,58,64,70,74,75,76,75,74,52,51,51,
54,55,54,55,52,52,53,55,55,84,82,81,82,81,82,85,92,96,96,96,92,85,83,84,83,85,92,93,92]]).T
'''
BASE_LANDMARKS_GBDT=np.matrix(
      [[ 12,  12,  14,  15,  19,  27,  37,  49,  63,  78,  90, 100, 108,
        112, 113, 115, 115,  21,  28,  37,  46,  55,  72,  81,  90,  99,
        106,  63,  64,  63,  64,  54,  58,  63,  68,  72,  34,  39,  46,
         50,  45,  39,  77,  82,  88,  93,  89,  83,  47,  53,  59,  63,
         68,  73,  79,  74,  68,  63,  58,  52,  50,  59,  63,  68,  76,  68,  63,  58],
       [ 39,  54,  68,  83,  97, 110, 120, 129, 131, 129, 121, 110,  97,
         83,  69,  55,  40,  36,  31,  30,  32,  35,  34,  31,  30,  31,
         36,  43,  54,  65,  75,  79,  82,  84,  82,  79,  41,  38,  38,
         43,  43,  43,  42,  38,  38,  41,  43,  43,  97,  96,  95,  97,
         95,  96,  97, 102, 104, 105, 104, 102,  98,  99, 100,  99,  98,  99, 100,  99]]).T
BASE_LANDMARKS_MTCNN=np.matrix([(48, 97), (78, 97), (41, 40), (86, 40), (64, 75)])

def predict(img, mode='gbdt'):
    if mode == 'mtcnn':
        return predict_mtcnn(img)
    landmarks = []
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        shape = predictor(img, d) 
        landmark = np.array([[p.x, p.y] for p in shape.parts()])
        landmarks.append(landmark)
    return dets, landmarks
        
def predictSingle(img,mode='gbdt'):
    if mode == 'mtcnn':
        return predict_mtcnn(img)[1][0]
    dets = detector(img, 1)
    landmark = np.array([])
    for i, d in enumerate(dets):
        shape = predictor(img, d) 
        landmark = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return landmark

def predict_mtcnn(img):
    def serialize_dict(d):
        rect = d['box']
        k = d['keypoints']
        landmarks = [k['mouth_left'], k['mouth_right'], k['left_eye'], k['right_eye'], k['nose']]
        landmarks = np.array(landmarks)
        return rect, landmarks
    res_ = d.detect_faces(img)  
    rects = []
    lms = []
    for r in res_:
        r1, r2 = serialize_dict(r)
        rects.append(r1)
        lms.append(r2)
    return rects, lms
    
def transformation_from_points(points2,points1=BASE_LANDMARKS_GBDT[ALIGN_POINTS]):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                         c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
    
def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
    
def align(img,landmarks,landmark_default=BASE_LANDMARKS_GBDT[ALIGN_POINTS]):
    #From Repo: matthewearl/faceswap
    M = transformation_from_points(landmarks,landmark_default)
    aligned_img = warp_im(img, M, (128,128,3))
    return aligned_img

def sf_gbdt(img):
    landmarks=[]
    rects=[]
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        rects.append([[d.left(),d.top()],[d.right(),d.bottom()]])
        shape = predictor(img, d) 
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        landmarks.append(landmark)
    faces=[]
    for lm in landmarks:
        i=align(img,lm[ALIGN_POINTS])
        faces.append(i)
    return rects, faces

def sf_mtcnn(img):
    def serialize_dict(d):
        rect = d['box']
        k = d['keypoints']
        landmarks = [k['mouth_left'], k['mouth_right'], k['left_eye'], k['right_eye'], k['nose']]
        landmarks = np.matrix(landmarks)
        return rect, landmarks
    res_ = d.detect_faces(img)  
    rects = []
    lms = []
    for r in res_:
        r1, r2 = serialize_dict(r)
        rects.append(r1)
        lms.append(r2)
    faces=[]
    for lm in lms:
        i=align(img,lm,BASE_LANDMARKS_MTCNN)
        faces.append(i)
    return rects, faces

def separateFace(img,mode='gbdt'):
    if mode == 'gbdt':
        return sf_gbdt(img)
    elif mode == 'mtcnn':
        return sf_mtcnn(img)
    else:
        return [],[]
    
def contrast(img,resize=1.):
    img=io.imread(img)
    size=np.array(img.shape,dtype=np.float64)[:2]
    size=size*resize
    size=np.array(size,dtype=np.int32)
    size=np.array([size[1],size[0]])
    size=tuple(size)
    img=cv2.resize(img,size)
    mode='mtcnn'
    time_start=time.time()
    rects,faces=predict(img,mode=mode)
    time_end=time.time()
    time_cost1=time_end-time_start
    
    mode='gbdt'
    time_start=time.time()
    rects,faces=predict(img,mode=mode)
    time_end=time.time()
    time_cost2=time_end-time_start
    
    return time_cost2, time_cost1

c_list=['1.jpg','2.jpg','3.jpg']
r_list=np.arange(0.5,1.6,0.1)

if __name__=='__main__':
    for img in c_list:
        times=[[],[]]
        for i in r_list:
            t1,t2=contrast(img,i)
            times[0].append(t1)
            times[1].append(t2)
            print('Time Cost: %.4fs(MTCNN) - %.4fs(GBDT) (resize %.2f)'%(t2,t1,i))
        plt.title('Contrastion on {}'.format(img))
        plt.xlabel('zooming')
        plt.ylabel('time cost/s')
        a,=plt.plot(r_list,np.array(times[1]),linewidth=1,label='MTCNN')
        b,=plt.plot(r_list,np.array(times[0]),linewidth=1,label='GBDT')
        plt.legend(handles=[a,b,],labels=['MTCNN','GBDT'],loc='best')
        plt.show()
        
        
    img=io.imread("1.jpg")
    mode='mtcnn'
    dets,landmarks=predict(img,mode=mode)
    rects,faces=separateFace(img,mode=mode) 
    #for f in faces:
    #    plt.imshow(f)
    #    plt.show() 
    #for i, d in enumerate(dets):
    #    cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)    
    for landmark in landmarks:
        for l in landmark:
            #print(l)
            cv2.rectangle(img,(l[0]-1,l[1]-1),(l[0]+1,l[1]+1),(0,0,255),2)
    plt.imshow(img)
    plt.show()
    
    


