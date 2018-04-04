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
BASE_LANDMARKS=np.matrix(
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

def predict(img):
    landmarks=[]
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        shape = predictor(img, d) 
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
        landmarks.append(landmark)
    return dets, landmarks
        
def predictSingle(img):
    dets = detector(img, 1)
    landmark=np.array([])
    for i, d in enumerate(dets):
        shape = predictor(img, d) 
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
    return landmark
    
def transformation_from_points(points2,points1=BASE_LANDMARKS[ALIGN_POINTS]):
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
    
def align(img,landmarks):
    #From Repo: matthewearl/faceswap
    M = transformation_from_points(landmarks[ALIGN_POINTS])
    aligned_img = warp_im(img, M, (128,128,3))
    return aligned_img

def separateFace(img):
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
        i=align(img,lm)
        faces.append(i)
    return rects, faces

if __name__=='__main__':
    img=io.imread("1.jpg")
    dets,landmarks=predict(img)
    rects,faces=separateFace(img)
    '''
    for i, d in enumerate(dets):print(i,d)
    for landmark in landmarks:
        plt.imshow(align(img,landmark))
        plt.show()  
    '''
    for f in faces:
        plt.imshow(f)
        plt.show() 
    for r in rects:print(r)
    
    for i, d in enumerate(dets):
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)    

    for landmark in landmarks:
        for l in landmark:
            cv2.rectangle(img,(l[:,0],l[:,1]),(l[:,0]+1,l[:,1]+1),(0,0,255),1)
    plt.imshow(img)
    plt.show()


