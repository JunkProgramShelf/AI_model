"""
Written by JunkProgramShelf
"""
import cv2
import os,glob
import sys
#import torch as th
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from sklearn import model_selection
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Flatten,Conv2D,MaxPooling2D,Dropout,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#np.random.seed(1)

"""
#setting GPU function
def get_device(gpuid):
	if gpuid==True:
		return th.device("cuda")
	else:
		return th.device("cpu")
"""
#Error function
def err(e):
	print("Program Error/code:"+str(e))

#input data define
file_size=100
input_layer=(file_size,file_size,3)

#Traning data class 
dir_path="./Image/Training/"
learn_img_class=[f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
print(learn_img_class)

class_num=len(learn_img_class)

#GPU setting
#device=get_device(th.cuda.is_available())

#data array
X_learn=[]#data
Y_learn=[]#index 
X_test=[]#data
Y_test=[]#index 
count=0

X=[]
Y=[]



#learning data file reading roop
for index,class_label in enumerate(learn_img_class):
	photo_dir=dir_path+class_label+"/*.jpg"
	files=sorted(glob.glob(photo_dir))
	for i,file in enumerate(files):
		if not os.path.exists(file):
			err(file+" is not exist")
			break
		Img=Image.open(file)
		Img=Img.resize((file_size,file_size))
		Img=Img.convert("RGB")
		data=np.array(Img)
		
		X.append(data)
		Y.append(index)
		
		#loop break
		#if i>=500:
		#	break
			



#numpy array excange
X=np.array(X)
Y=np.array(Y)

#Normalization
X=X.astype('float32')/255.


#train and test data split
x_train,x_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=0.3)

#one-hot
y_train=utils.to_categorical(y_train,class_num)
y_test=utils.to_categorical(y_test,class_num)



#print(X_learn.shape)
#print(X_learn)
#print(Y_learn.shape)
#print(Y_learn)


#coding AI's model
filters=16
karnel_size=(4,4)
model=tf.keras.Sequential()
model.add(Conv2D(filters,karnel_size,padding='same',input_shape=input_layer,activation='relu'))
model.add(Conv2D(filters,karnel_size,activation='relu'))
model.add(MaxPooling2D((2,2)))

filters=2*filters
model.add(Dropout(0.10))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(MaxPooling2D((2,2)))

filters=2*filters
model.add(Dropout(0.10))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(MaxPooling2D((2,2)))

filters=2*filters
model.add(Dropout(0.10))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(MaxPooling2D((2,2)))
"""
filters=2*filters
model.add(Dropout(0.30))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(MaxPooling2D((2,2)))


filters=2*filters
model.add(Dropout(0.20))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(Conv2D(filters,karnel_size,padding='same',activation='relu'))
model.add(MaxPooling2D((2,2)))
"""
model.add(Flatten())
model.add(Dropout(0.20))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(class_num,activation='softmax'))

model.summary()



#Learning
epoch=30
batch=50

start_time=time.time()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,weight_decay=1e-7),loss='mean_squared_error',metrics=['accuracy'])
history=model.fit(x_train,y_train,batch,epoch,verbose=0,validation_data=(x_test,y_test))
training_time=time.time()-start_time

#
model.save('./Learned_model.h5')
print("time::::::"+str(training_time))


#score
#Training data
score=model.evaluate(x_train,y_train,0)
print("loss *** {:.6f}".format(score[0]))
print("accuracy *** {:.6f}".format(score[1]))
#Test data
score2=model.evaluate(x_test,y_test,0)
print("loss *** {:.6f}".format(score2[0]))
print("accuracy *** {:.6f}".format(score2[1]))



#plot to accuracy in graph
plt.figure()
plt.title("accuracy_in_epoch"+str(epoch))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["learn","test"])
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.grid()
plt.savefig("./Image/learned_acc"+".png")

#plot to loss in graph
plt.figure()
plt.title("loss_in_epoch"+str(epoch))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["learn","test"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.grid()
plt.savefig("./Image/learned_loss"+".png")


