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

root='F:\\Datasets'
#root='C:\\Users\\zhaoj\\Documents\\Datasets'
train_dir_vap=root+'\\VAPRBGD\\train\\'
val_dir_vap=root+'\\VAPRBGD\\val\\'
train_dir_vgg=root+'\\VGGFACE\\train\\'
val_dir_vgg=root+'\\VGGFACE\\val\\'
train_dir_vgg2=root+'\\VGGFACE2\\train\\'
val_dir_vgg2=root+'\\VGGFACE2\\val\\'
cb_dir='callbacks.h5'

print('Recognition Networks Loaded.')

#Contrastive Loss
def callbacks():
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch>80:lr*=0.5e-3
        elif epoch>60:lr*=1e-3
        elif epoch>40:lr*=1e-2
        elif epoch>20:lr*=1e-1
        print('Learning rate: ', lr)
        return lr
    
    checkpoint = ModelCheckpoint(filepath=cb_dir,monitor='val_acc',verbose=1,save_best_only=True)
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
            model.fit_generator(gen, steps_per_epoch=30, epochs=80, validation_data = val_gen, validation_steps=20, callbacks=callbacks())
        except KeyboardInterrupt:
            print('KeyboardInterrupt Received. Weights Saved.')
        finally:
            model.save_weights(savepath)
    return model

def mn_vgg2(load=1,opt='sgd',savepath='mn2.h5'):
    gen=data_loader.gen_vgg2
    val_gen=data_loader.val_gen_vgg2
    model=models.MobileNet_FT(opt=opt)
    
    if load==1:
        print('Loading weights...')
        model.load_weights(savepath)
    else:
        try:
            #pass
            model.fit_generator(gen, steps_per_epoch=30, epochs=100, validation_data = val_gen, validation_steps=20, callbacks=callbacks())
        except KeyboardInterrupt:
            print('KeyboardInterrupt received. Weights saved.')
        finally:
            model.save_weights(savepath)
            
    print('Fine_Tuned MobileNet loaded, using VGGFace2 dataset.')
    return model

def evaluate_cl(model,val_dir,single,form='jpg',shape=(1,128,128,3),time=5000):
    cp,cn,cs=[],[],[]
    val_dir=glob.glob(val_dir+'*')
    
    for i in range(time):
        pathp=data_loader.get_dir(val_dir,'positive',form)
        pathn=data_loader.get_dir(val_dir,'negative',form)
        cop = data_loader.create_pair_rgb(pathp,single,form)
        c1=model.predict([cop[0].reshape(shape), cop[1].reshape(shape)])[0,0]
        cop = data_loader.create_pair_rgb(pathn,single,form)
        c2=model.predict([cop[0].reshape(shape), cop[1].reshape(shape)])[0,0]
        if i%50==0:print('Group %d: Positive: %.3f - Negative: %.3f'%(i+1,c1,c2))
        
        cp.append(c1)
        cn.append(c2)
        cs.append(abs(c2-c1))
        
    cs.sort()
    cp.sort()
    cn.sort(reverse=True)

    plt.scatter(range(time),cp,s=1e-1)
    plt.scatter(range(time),cn,s=1e-1)
    plt.scatter(range(time),cs,s=3e-2)
    plt.show()
    
#LMCL
def callbacks_lmcl(opt='adam'):
    def lrs_sgd(epoch):
        lr=1e-2
        if epoch>210:lr*=5e-5
        elif epoch>170:lr*=1e-4
        elif epoch>130:lr*=5e-4
        elif epoch>90:lr*=1e-3
        elif epoch>55:lr*=1e-2
        elif epoch>25:lr*=1e-1
        print('Learning rate: ', lr)
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
        print('Learning rate: ', lr)
        return lr
    lr_schedule={'adam':lrs_adam,'sgd':lrs_sgd}
    lr_scheduler=LearningRateScheduler(lr_schedule[opt]) 
    board=TensorBoard(log_dir='./logs')
    checkpoint=ModelCheckpoint(filepath=cb_dir,monitor='val_acc',verbose=1,save_best_only=True)
    stopping=EarlyStopping(patience=10,verbose=0,mode='auto')

    lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=15,epsilon=0.001,min_lr=1e-9)
    
    return [
            #checkpoint,
            lr_reducer,
            #lr_scheduler,
            board
            #stopping
           ]

def mn_vgg2_lmcl(load=1,opt='adam',savepath='mn_lmcl.h5',units=500,preload=None):
    
    gen=data_loader.singleGenerator(train_dir_vgg2,count=units)
    val_gen=data_loader.singleGenerator(train_dir_vgg2,select='val',count=units)
    
    if load!=1:
        model=models.MobileNet_LMCL(opt=opt,units=units)
        if isinstance(preload,str):
            model.load_weights(preload)
        try:
            model.fit_generator(gen, steps_per_epoch=32, epochs=5000, 
                                validation_data = val_gen, validation_steps=10, 
                                callbacks=callbacks_lmcl(opt))
        except KeyboardInterrupt:
            print('KeyboardInterrupt received. Weights saved.')
        finally:
            model.save_weights(savepath)
    
    print('Loading weights...')
    model=models.MobileNet_LMCL(opt=opt,output_fc=True,units=units)
    model.load_weights(savepath,by_name=True)
    print('Fine_Tuned MobileNet loaded, using VGGFace2 dataset and LMCL loss.')
    
    return model

def main():
    '''
    model=mn_vgg2(1)
    evaluate_cl(model,val_dir_vgg2,data_loader.create_single_VGGFACE)
    model=mn_vgg2(1,opt='adam',savepath='mn1.h5')
    evaluate_cl(model,val_dir_vgg2,datna_loader.create_single_VGGFACE)
    '''
    #model=mn_vgg2_lmcl(0)
    model=mn_vgg2_lmcl(0,savepath='mn_lmcl_adam.h5',preload='mn_lmcl_sgd.h5')
    
if __name__=='__main__':
    main()
