import cv2
import os
import numpy as np
import re

dataset_path = '../dataset'
detector_conf_path = '../face_detect/cv_haarcascade_config/haarcascade_frontalface_default.xml'

if __name__ == '__main__':
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(detector_conf_path)
    
    image_names = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
                   
    samples, ids = [], []
    for name in image_names:
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        face_id = int(re.findall('\d+', name)[0])
        
        samples.append(img)
        ids.append(face_id)

    recognizer.train(samples, np.array(ids))
    recognizer.write('LBPH_recognizer.yml')
    print('%d faces trained.' %(len(np.unique(ids))))                   

