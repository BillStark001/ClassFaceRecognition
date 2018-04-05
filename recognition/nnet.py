# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 20:29:27 2018
@author: BillStark001
"""

#LOCAL
try:
    import models
    import data_loader
except ImportError as e:
    import recognition.models as models
    import recognition.data_loader as data_loader
#NOT LOCAL
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

root='F:\\Datasets'
#root='C:\\Users\\zhaoj\\Documents\\Datasets'
train_dir_vap=root+'\\VAPRBGD\\train\\'
val_dir_vap=root+'\\VAPRBGD\\val\\'
train_dir_vgg=root+'\\VGGFACE\\train\\'
val_dir_vgg=root+'\\VGGFACE\\val\\'
cb_dir=root+'\\callbacks'

print('Recognition Networks Loaded.')

def callbacks():
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch>100:lr*=0.5e-3
        elif epoch>70:lr*=1e-3
        elif epoch>35:lr*=1e-2
        elif epoch>15:lr*=1e-1
        print('Learning rate: ', lr)
        return lr
    
    checkpoint = ModelCheckpoint(filepath=cb_dir,monitor='val_loss',verbose=1,save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
    
    return [checkpoint,lr_reducer,lr_scheduler]

def sn_vap(load=1,savepath='sn.h5'):
    gen=data_loader.gen_vap
    val_gen=data_loader.val_gen_vap
    model=models.SqueezeNet()
    if load==1:
        model.load_weights(savepath)
    else:
        model.fit_generator(gen, steps_per_epoch=30, epochs=50, validation_data = val_gen, validation_steps=20)
        model.save(savepath)
    return model

def mn_vgg(load=1,savepath='mn.h5'):
    gen=data_loader.gen_vgg
    val_gen=data_loader.val_gen_vgg
    model=models.MobileNet_FT()
    print('MobileNet_Fine_Tuned Loaded.')
    if load==1:
        model.load_weights(savepath)
        print('Weights Loaded.')
    else:
        try:
            model.fit_generator(gen, steps_per_epoch=40, epochs=30, validation_data = val_gen, validation_steps=30, callbacks=callbacks())
        except KeyboardInterrupt:
            print('KeyboardInterrupt Received. Weights Saved.')
        finally:
            model.save_weights(savepath)
    return model

def xc_vgg(load=1,savepath='xc.h5'):
    gen=data_loader.gen_vgg
    val_gen=data_loader.val_gen_vgg
    model=models.XCeption_FT()
    if load==1:
        model.load_weights(savepath)
    else:
        try:
            model.fit_generator(gen, steps_per_epoch=20, epochs=20, validation_data = val_gen, validation_steps=20)
        except KeyboardInterrupt:
            print('KeyboardInterrupt Received. Weights Saved.')
        finally:
            model.save_weights(savepath)
    return model

def evaluate_vap(model,time=100):
    cp,cn,cs=[],[],[]
    
    for i in range(time):
        print(i+1)
        cop = data_loader.create_positive_rgb(val_dir_vap,data_loader.create_single_VAPRGBD,'bmp')
        c1=model.predict([cop[0].reshape((1,112,112,3)), cop[1].reshape((1,112,112,3))])[0,0]
        
        cop = data_loader.create_negative_rgb(val_dir_vap,data_loader.create_single_VAPRGBD,'bmp')
        c2=model.predict([cop[0].reshape((1,112,112,3)), cop[1].reshape((1,112,112,3))])[0,0]
    
        cp.append(c1)
        cn.append(c2)
        cs.append(abs(c2-c1))
        cs.sort()
        cp.sort()
        cn.sort()
        
    plt.plot(range(time),cp)
    plt.plot(range(time),cn)
    plt.show()
    plt.scatter(range(time),cs,s=1)
    plt.show()
    
def evaluate_vgg(model,time=100):
    cp,cn,cs=[],[],[]
    
    for i in range(time):
        print(i+1)
        cop = data_loader.create_positive_rgb(val_dir_vgg,data_loader.create_single_VGGFACE)
        c1=model.evaluate([cop[0].reshape((1,128,128,3)), cop[1].reshape((1,128,128,3))], np.array([0.]))
        
        cop = data_loader.create_negative_rgb(val_dir_vgg,data_loader.create_single_VGGFACE)
        c2=model.evaluate([cop[0].reshape((1,128,128,3)), cop[1].reshape((1,128,128,3))], np.array([1.]))
    
        cp.append(c1)
        cn.append(c2)
        cs.append(abs(c2-c1))
        cs.sort()
        cp.sort()
        cn.sort()
    
    plt.plot(range(time),cp)
    plt.plot(range(time),cn)
    plt.show()
    plt.scatter(range(time),cs,s=1)
    plt.show()
    
def main():
    model=mn_vgg(0)
    evaluate_vgg(model)
    
if __name__=='__main__':
    main()
