# -*- coding: utf-8 -*-
'''
Created on Fri Feb 23 15:38:56 2018
@author: BillStark001
'''
import os
#LOCAL
import vstream.cv as vst #Video Stream using opencv
import face_detect.cv as detect #Now not used
import alignment.main as align #Alignment using dlib
try:
    assert rec
except:
    wkdir=os.getcwd()
    import recognition.interface as rec
    os.chdir(wkdir)
#Recognition
print('Import Finished.')
#OTHERS
import numpy as np
import cv2

#root='F:\\Datasets\\TestFace\\'
root='C:\\Users\\zhaoj\\Documents\\Datasets\\TestFace\\'
data_dir=root+'\\registered\\'
test_dir=root+'\\test\\'

#print(os.getcwd())
embed_path='recognition\\embed.pkl'

Input=vst.VStream((1,0),(1280,720))
font = cv2.FONT_HERSHEY_SIMPLEX

'''
def detectAndDraw(image=Input.getCurrentIMG()):
    coors=detect.detectFace(image)
    image=detect.drawFaceCoor(image,coors)
    imgs=np.array(detect.splitbyCoor(image,coors,bias=20))
    if imgs.shape[0]==0:return image
    
    for i in imgs
    
    if not imgs.shape[0]==0:
        for i in range(imgs.shape[0]):
            img=np.array(imgs[i],dtype='uint8')
            landmark=np.array(align.predictSingle(img))
            if not len(landmark)==0:
                for l in landmark:
                    l[0]=l[0]+coors[i,0]-20
                    l[1]=l[1]+coors[i,1]-20
                    cv2.rectangle(image,(l[0],l[1]),(l[0]+1,l[1]+1),(255,0,0),1)
            
            aligned_img=align.align(image,np.matrix(landmark))
            cv2.imshow('Aligned',aligned_img)

    #detect.saveFaces(detect.splitbyCoor(image,coors,bias=20))
    
    return image
'''
def detectAndDraw(image=Input.getCurrentIMG()):
    
    rects,faces=align.separateFace(image)
    for i in range(len(rects)):
        l,f=rects[i],faces[i]
        x,y,w,h=l[0][0],l[0][1],l[1][0],l[1][1]
        cv2.rectangle(image,(x,y),(w,h),(0,255,0),1)
        bias=5
        
        #maxn,score,ans=rec.enumPath(f,data_dir,mode='cons')
        maxn,score,ans=rec.enumPath(f,rec.load_embed(embed_path),mode='lmcl')
        print('Face {}: {}, score={}.'.format(i,maxn,score))
        print(np.array(ans))
        
        cv2.putText(image,maxn,(x+bias,y-bias),font,1,(255,0,255),2)
        cv2.putText(image,str(score),(x+bias,h-bias),font,1,(255,255,0),1)
    if faces!=[]:
        cv2.imshow('Aligned',faces[0])
    
    return image

def showDetect(Wait=10,Key='q'):
    while(1):
        try:
            image=Input.getCurrentIMG()
            cv2.imshow('Capturing',detectAndDraw(image))
            if cv2.waitKey(Wait)&0xFF==ord(Key):
                print('Abort key\'{}\' Received. Recognition Aborted.'.format(Key))
                break
        except KeyboardInterrupt:
            print('KeyboardInterrupt Received. Recognition Aborted.')
            break
    
if __name__=='__main__':
    Input.startCapture()
    showDetect()#Input.showCapture(source=detectAndDraw())
    Input.stopCapture()