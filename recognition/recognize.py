import cv2
import os
import numpy as np

detector_conf_path = '../face_detect/cv_haarcascade_config/haarcascade_frontalface_default.xml'
recognizer_conf_path = 'LBPH_recognizer.yml'

font = cv2.FONT_HERSHEY_SIMPLEX

name_dict = {2333: 'hzx', 6666: 'zjn', 1001: 'gyp', 1919: 'jcc', 2132: 'syl'}

if __name__ == '__main__':
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(recognizer_conf_path)
    
    detector = cv2.CascadeClassifier(detector_conf_path)
    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            face_id, confidence = recognizer.predict(gray[y: y + h, x: x + w])
            print(face_id, confidence)
            
            cv2.putText(img, name_dict[face_id], (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
            
        cv2.imshow('capture', img)
        
        key = cv2.waitKey(100) & 0xff
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
        