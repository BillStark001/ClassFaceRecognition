'''
What the hell????
'''

#lOCAL
import vstream.cv as vst
import face_detect.cv as detect
import alignment.main as align
#NOT LOCAL
import cv2
import os

root='C:\\Users\\zhaoj\\Documents\\Datasets\\TestFace\\'
data_dir=root+'\\registered\\'
save_dir=root+'\\test\\'

casc_path_default = 'cv_haarcascade_config/haarcascade_frontalface_default.xml'
save_threshold = 8

Input=vst.VStream((1,0),(1280,720))

def main():
    
    Input.startCapture()
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    
    face_name = input('input face name:')
    count = 0
    
    while True:
        img = Input.getCurrentIMG()
        rects, faces = align.separateFace(img)
        
        for l,f in zip(rects,faces):

            count += 1
            cv2.imwrite(save_dir+'{}.{}.jpg'.format(face_name,count),f)
            x,y,w,h=l[0][0],l[0][1],l[1][0],l[1][1]
            cv2.rectangle(img,(x,y),(w,h),(0,255,0),1)
            cv2.imshow('capture', img)
        
        if count > save_threshold:
            break
        
        key = cv2.waitKey(100) & 0xff
        if key == ord('q'):
            break
    
    Input.stopCapture()
    
if __name__ == '__main__':
    main()
        