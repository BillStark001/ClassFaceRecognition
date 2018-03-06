import cv2
import os

casc_path_default = 'cv_haarcascade_config/haarcascade_frontalface_default.xml'
save_threshold = 64

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    
    face_id = int(input('input face id:'))
    count = 0
    
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        classifier = cv2.CascadeClassifier(casc_path_default)
        faces = classifier.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            count += 1
            cv2.imwrite('../dataset/user.%d.%d.jpg' %(face_id, count), gray[y: y + h, x: x + w])
            cv2.imshow('capture', img)
        
        if count > save_threshold:
            break
        
        key = cv2.waitKey(100) & 0xff
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
        