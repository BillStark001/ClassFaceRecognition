# -*- coding: utf-8 -*-
'''
Created on Mon Mar  5 20:29:27 2018
@author: BillStark001
'''

#LOCAL
try:
    import models
    import data_loader
except ImportError as e:
    import recognition.models as models
    import recognition.data_loader as data_loader
#NOT LOCAL
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam, SGD

root='F:\\Datasets'
#root='C:\\Users\\zhaoj\\Documents\\Datasets'
train_dir_vgg2=root+'\\VGGFACE2\\train\\'
val_dir_vgg2=root+'\\VGGFACE2\\val\\'
cb_dir='callbacks.h5'

print('Recognition Networks Loaded.')

def callbacks(opt='adam'):
    if isinstance(opt, Adam):
        opt = 'adam'
    elif not isinstance(opt, str):
        opt = 'sgd'
    def lrs_sgd(epoch):
        lr=1e-2
        if epoch>160:lr*=5e-5
        elif epoch>130:lr*=1e-4
        elif epoch>100:lr*=5e-4
        elif epoch>80:lr*=1e-3
        elif epoch>50:lr*=1e-2
        elif epoch>20:lr*=1e-1
        print('Learning rate: ',lr)
        return lr
    def lrs_adam(epoch):
        lr=1e-3
        if epoch>225:lr*=1e-5
        elif epoch>180:lr*=5e-5
        elif epoch>145:lr*=1e-4
        elif epoch>110:lr*=5e-4
        elif epoch>80:lr*=1e-3
        elif epoch>50:lr*=1e-2
        elif epoch>25:lr*=1e-1
        print('Learning rate: ',lr)
        return lr
    lr_schedule={'adam':lrs_adam,'sgd':lrs_sgd}
    lr_scheduler=LearningRateScheduler(lr_schedule[opt]) 
    board=TensorBoard(log_dir='./logs')
    stopping=EarlyStopping(patience=10,verbose=0,mode='auto')

    lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=7,epsilon=0.001,min_lr=1e-9)
    
    return_schedule={'sgd':[lr_reducer, board],'adam':[lr_reducer, board]}
    return return_schedule[opt]
    
#Contrastive Loss

def mn_vgg2(load=1,opt='sgd',savepath='mn2.h5'):
    gen=data_loader.gen_vgg2
    val_gen=data_loader.val_gen_vgg2
    
    if load!=1:
        model=models.MobileNet_FT(opt=opt)
        try:
            model.fit_generator(gen,steps_per_epoch=48,epochs=100,
                                validation_data=val_gen,validation_steps=24,
                                callbacks=callbacks(opt))
        except KeyboardInterrupt:
            print('KeyboardInterrupt received. Weights saved.')
        finally:
            model.save_weights(savepath)
    
    print('Loading weights...')
    model=models.MobileNet_FT(opt=opt,output_fc=True)
    model.load_weights(savepath,by_name=True)
    print('Fine_Tuned MobileNet loaded, using Contrastive loss.')
    
    return model

def eucilidian_distance(u, v):
    ans = u - v
    ans = np.sqrt(np.dot(ans, ans.T))
    return ans

def evaluate_cl(model,val_dir,single,form='jpg',shape=(1,128,128,3),time=250,acc=1000):
    cp,cn,cs=[],[],[]
    val_dir=glob.glob(val_dir+'*')
    
    for i in range(time):
        pathp=data_loader.get_dir(val_dir,'positive',form)
        pathn=data_loader.get_dir(val_dir,'negative',form)
        cop=data_loader.create_pair_rgb(pathp,single,form)
        c11=model.predict([cop[0].reshape(shape)])
        c12=model.predict([cop[1].reshape(shape)])
        c1=eucilidian_distance(c11,c12)
        cop=data_loader.create_pair_rgb(pathn,single,form)
        c21=model.predict([cop[0].reshape(shape)])
        c22=model.predict([cop[1].reshape(shape)])
        c2=eucilidian_distance(c21,c22)
        if (i+1)%50==0:
            print('Group %d: Positive: %.3f - Negative: %.3f'%(i+1,c1,c2))
        
        cp.append(c1)
        cn.append(c2)
        cs.append(abs(c2-c1))
        
    #cs.sort()
    cp.sort()
    cn.sort(reverse=True)
    cp=np.array(cp)*acc
    cp=np.array(cp,dtype=np.int32)
    cn=np.array(cn)*acc
    cn=np.array(cn,dtype=np.int32)
    
    plt.scatter(range(cp),cp,s=1e-0)
    plt.scatter(range(cn),cn,s=1e-0)
    plt.show()
    
#LMCL
def data_generator(path, target_size=(112, 112), batch_size=128):
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
            horizontal_flip=True,
            preprocessing_function=lambda x: (x - 127.5) / 128,
    )
    train_gen = datagen.flow_from_directory(path, target_size=target_size, batch_size=batch_size)
    # val_gen = datagen.flow_from_directory(path, target_size=target_size, batch_size=batch_size)
    # val_gen is now identical with train_gen
    return train_gen

def mn_vgg2_lmcl(load=1, version=1, opt='adam', savepath='mn_lmcl.h5', classes=1000, preload=None, epochs=200):
    gen = data_loader.singleGenerator(train_dir_vgg2, count=classes, separate=1, batch_size=(16,3))
    val_gen = data_loader.singleGenerator(val_dir_vgg2, count=classes, separate=1, batch_size=(8,3))
    if load!=1:
        if version == 1:
            model = models.MobileNet_LMCL(opt=opt, classes=classes)
        else:
            model = models.MobileNetV2_LMCL(opt=opt, classes=classes)
        if isinstance(preload, str):
            model.load_weights(preload, by_name=True)
        try:
            model.fit_generator(gen, steps_per_epoch=36, epochs=epochs,
                                validation_data=val_gen, validation_steps=6,
                                callbacks=callbacks(opt))
        except KeyboardInterrupt:
            print('KeyboardInterrupt received. Weights saved.')
        finally:
            model.save_weights(savepath)
    
    print('Loading weights...')
    if version == 1:
        model = models.MobileNet_LMCL(opt=opt, output_fc=True, classes=classes)
    else:
        model = models.MobileNetV2_LMCL(opt=opt, output_fc=True, classes=classes)
    model.load_weights(savepath, by_name=True)
    print('Fine_Tuned MobileNet loaded, using LMCL loss.')
    
    return model

def train_lmcl_adam():
    adam_4 = Adam(lr=1e-4)
    #model = mn_vgg2_lmcl(0, opt='adam', version=2, epochs=50, classes=125, savepath='mn_lmcl_125.h5')
    #model = mn_vgg2_lmcl(0, opt='adam', version=2, epochs=100, classes=250, savepath='mn_lmcl_250.h5', preload='mn_lmcl_125.h5')
    #model = mn_vgg2_lmcl(0, opt=adam_4, version=2, epochs=150, classes=500, savepath='mn_lmcl_500.h5', preload='mn_lmcl_250.h5')
    model = mn_vgg2_lmcl(0, opt='adam', version=2, epochs=200, classes=1000, savepath='mn_lmcl_1000.h5', preload='mn_lmcl_500.h5')
    return model

def train_lmcl_sgd():
    #sgd_4 = SGD(lr=3e-4)
    #model = mn_vgg2_lmcl(0, opt='sgd', version=2, epochs=50, classes=125, savepath='mns_lmcl_125.h5')
    #model = mn_vgg2_lmcl(0, opt='sgd', version=2, epochs=100, classes=250, savepath='mns_lmcl_250.h5')
    #model = mn_vgg2_lmcl(0, opt='sgd', version=2, epochs=150, classes=500, savepath='mns_lmcl_500.h5', preload='mns_lmcl_250.h5')
    #model = mn_vgg2_lmcl(0, opt='sgd', version=2, epochs=200, classes=1000, savepath='mns_lmcl_1000.h5', preload='mns_lmcl_500.h5')
    model = mn_vgg2_lmcl(0, opt='sgd', version=2, epochs=200, classes=768, savepath='mns_lmcl_768.h5', preload='mn_lmcl_500.h5')
    return model

def train_cont():
    model = mn_vgg2(0)
    #model = mn_vgg2(0, opt='adam', savepath='mn1.h5')
    evaluate_cl(model, val_dir_vgg2, data_loader.create_single_VGGFACE)
    return model

def main():
    model = train_lmcl_sgd()
    
if __name__=='__main__':
    main()
