# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 20:29:27 2018
@author: BillStark001
"""

#LOCAL
import models
import data_loader
#NOT LOCAL
import numpy as np

root='C:\\Users\\zhaoj\\Documents\\Datasets'
train_dir=root+'\\VAPRBGD\\train\\'
val_dir=root+'\\VAPRBGD\\val\\'

gen=data_loader.gen_vap
val_gen=data_loader.val_gen_vap

model=models.SqueezeNet()
outputs=model.fit_generator(gen, steps_per_epoch=30, epochs=50, validation_data = val_gen, validation_steps=20)

cop = data_loader.create_positive_rgb(val_dir)
model.evaluate([cop[0].reshape((1,112,112,3)), cop[1].reshape((1,112,112,3))], np.array([0.]))

cop = data_loader.create_negative_rgb(val_dir)
model.predict([cop[0].reshape((1,112,112,3)), cop[1].reshape((1,112,112,3))])
'''
if __name__=='__main__':
    print(model)
    
    img_path='../face_0.png'
    img=image.load_img(img_path,target_size=(224, 224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    
    print('Predicting......')
    preds=model.predict(img)
    print('Predicted:', decode_predictions(preds))
    '''