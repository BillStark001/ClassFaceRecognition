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
import glob
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
train_dir_vgg2=root+'\\VGGFACE2\\train\\'
val_dir_vgg2=root+'\\VGGFACE2\\val\\'
cb_dir=root+'\\callbacks'

print('Recognition Networks Loaded.')

def callbacks():
    def lr_schedule(epoch):
        lr = 1e-2
        if epoch>70:lr*=0.5e-3
        elif epoch>50:lr*=1e-3
        elif epoch>35:lr*=1e-2
        elif epoch>17:lr*=1e-1
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
    print('Fine_Tuned MobileNet loaded, using VGGFace2 datasets.')
    if load==1:
        model.load_weights(savepath)
        print('Weights loaded.')
    else:
        try:
            #pass
            model.fit_generator(gen, steps_per_epoch=30, epochs=80, validation_data = val_gen, validation_steps=20, callbacks=callbacks())
        except KeyboardInterrupt:
            print('KeyboardInterrupt received. Weights saved.')
        finally:
            model.save_weights(savepath)
    return model

def evaluate_cl(model,val_dir,single,form='jpg',shape=(1,128,128,3),time=500):
    cp,cn,cs=[],[],[]
    val_dir=glob.glob(val_dir+'*')
    
    for i in range(time):
        pathp=data_loader.get_dir(val_dir,'positive',form)
        pathn=data_loader.get_dir(val_dir,'negative',form)
        cop = data_loader.create_pair_rgb(pathp,single,form)
        c1=model.predict([cop[0].reshape(shape), cop[1].reshape(shape)])[0,0]
        cop = data_loader.create_pair_rgb(pathn,single,form)
        c2=model.predict([cop[0].reshape(shape), cop[1].reshape(shape)])[0,0]
        print('Group %d: Positive: %.3f - Negative: %.3f'%(i+1,c1,c2))
        
        cp.append(c1)
        cn.append(c2)
        cs.append(abs(c2-c1))
        
    cs.sort()
    cp.sort()
    cn.sort(reverse=False)

    plt.scatter(range(time),cp,s=1)
    plt.scatter(range(time),cn,s=1)
    plt.scatter(range(time),cs,s=0.3)
    plt.show()
    
def main():
    model=mn_vgg(0)
    evaluate_cl(model,val_dir_vgg,data_loader.create_single_VGGFACE)
    model=mn_vgg2()
    evaluate_cl(model,val_dir_vgg2,data_loader.create_single_VGGFACE)
    
if __name__=='__main__':
    main()
