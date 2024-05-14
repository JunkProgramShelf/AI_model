import cv2
import os,glob
import seaborn as sns
import sys
import pandas as pd
#import torch as th
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Input,Dense,Flatten,Conv2D,MaxPooling2D,Dropout,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#np.random.seed(1)

file_size=100

def get_device(gpuid):
	if gpuid==True:
		return th.device("cuda")
	else:
		return th.device("cpu")

#Error function
def err(e):
	print("Program Error/code:"+str(e))

#input data define
dir_path="./Image/Test/"
img_class=[f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
class_num=len(img_class)
file_num = 100
input_shape=(file_size,file_size,3)


#GPU setting
device=get_device(th.cuda.is_available())


X=[]
Y=[]


#test data file reading roop
for index,class_label in enumerate(img_class):
	photo_dir=file_path+class_label+"/*.jpg"
	files=sorted(glob.glob(photo_dir))
	for i,file in enumerate(files):
		if not os.path.exists(file):
			print(file+" is not exist")
			break
		Img=Image.open(file)
		Img=Img.resize((file_size,file_size))
		Img=Img.convert("RGB")
		data=np.array(Img)

		X.append(data)
		Y.append(index)
		if i >= file_num:
			break






X = np.array(X)
Y = np.array(Y)


#input data exchange
Y_t=to_categorical(Y,class_num)
X=X.astype("float32")/255.
Y_t=np.array(Y_t)

#loading model
model=load_model('Learned_model.h5')
score= model.evaluate(X,Y_t,verbose=0)

#print accuracy and loss
print("loss and accruracy",score)
#print("loss::::{}".format(score[0]))
#print("acc:::::{}".format(score[1]))


p_array=np.zeros((class_num,class_num))
pred =  model.predict(X)
#print(pred.shape)
#print(pred)
count=0

for x in range(class_num):
	for y in range(file_num):
		#print(str(x)+":"+str(y))
		p_array[x]+=pred[count]
		count+=1
	count+=1
	#print(p_array[x])
p_array=p_array/float(file_num)
print(p_array)

#output heatmap
plt.figure()
pd_data=pd.DataFrame(p_array,index=img_class,columns=img_class)
sns.heatmap(pd_data)
plt.show()


np.savetxt("saving_predict.txt",pred)

#(^o^)/