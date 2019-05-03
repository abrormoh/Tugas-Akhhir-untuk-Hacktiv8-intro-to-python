import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

sf_train = pd.read_csv(r'C:\Users\Abror Mohammad\Downloads/fifa_data.csv')
sf_train.drop(sf_train.columns[[0,1,3]],axis=1, inplace=True)
sf_train.loc[sf_train.Position == 'LAM','Position']=0
sf_train.loc[sf_train.Position == 'LF','Position']=0
sf_train.loc[sf_train.Position == 'LS','Position']=0
sf_train.loc[sf_train.Position == 'RW','Position']=0
sf_train.loc[sf_train.Position == 'RF','Position']=0
sf_train.loc[sf_train.Position == 'RS','Position']=0
sf_train.loc[sf_train.Position == 'LW','Position']=0
sf_train.loc[sf_train.Position == 'ST','Position']=0
sf_train.loc[sf_train.Position == 'CAM','Position']=1
sf_train.loc[sf_train.Position == 'CDM','Position']=1
sf_train.loc[sf_train.Position == 'CM','Position']=1
sf_train.loc[sf_train.Position == 'LCM','Position']=1
sf_train.loc[sf_train.Position == 'LDM','Position']=1
sf_train.loc[sf_train.Position == 'LM','Position']=1
sf_train.loc[sf_train.Position == 'RCM','Position']=1
sf_train.loc[sf_train.Position == 'RDM','Position']=1
sf_train.loc[sf_train.Position == 'RM','Position']=1
sf_train.loc[sf_train.Position == 'CB','Position']=2
sf_train.loc[sf_train.Position == 'LB','Position']=2
sf_train.loc[sf_train.Position == 'LCB','Position']=2
sf_train.loc[sf_train.Position == 'GK','Position']=3

sf_val = pd.read_csv(r'C:\Users\Abror Mohammad\Downloads/fifa_val.csv')
sf_val.drop(sf_val.columns[[0,1,3]],axis=1, inplace=True)
sf_val.loc[sf_val.Position == 'LAM','Position']=0
sf_val.loc[sf_val.Position == 'LF','Position']=0
sf_val.loc[sf_val.Position == 'LS','Position']=0
sf_val.loc[sf_val.Position == 'RW','Position']=0
sf_val.loc[sf_val.Position == 'RF','Position']=0
sf_val.loc[sf_val.Position == 'RS','Position']=0
sf_val.loc[sf_val.Position == 'ST','Position']=0
sf_val.loc[sf_val.Position == 'LW','Position']=0
sf_val.loc[sf_val.Position == 'CAM','Position']=1
sf_val.loc[sf_val.Position == 'CDM','Position']=1
sf_val.loc[sf_val.Position == 'CM','Position']=1
sf_val.loc[sf_val.Position == 'LCM','Position']=1
sf_val.loc[sf_val.Position == 'LDM','Position']=1
sf_val.loc[sf_val.Position == 'LM','Position']=1
sf_val.loc[sf_val.Position == 'RCM','Position']=1
sf_val.loc[sf_val.Position == 'RDM','Position']=1
sf_val.loc[sf_val.Position == 'RM','Position']=1
sf_val.loc[sf_val.Position == 'CB','Position']=2
sf_val.loc[sf_val.Position == 'LB','Position']=2
sf_val.loc[sf_val.Position == 'LCB','Position']=2
sf_val.loc[sf_val.Position == 'GK','Position']=3

train_data=sf_train.values
val_data = sf_val.values

# Use columns 2 to last as Input
train_x = train_data[:,2:]
val_x = val_data[:,2:]

# Use columns 1 as Output/Target (One-Hot Encoding)
train_y = to_categorical( train_data[:,1] )
val_y = to_categorical( val_data[:,1] )
# print(len(train_x))

inputs = Input(shape=(34,))
h_layer = Dense(10,activation='sigmoid')(inputs)
outputs = Dense(4,activation='softmax')(h_layer)
model = Model(inputs=inputs,outputs=outputs)
sgd = SGD (lr=0.001)
model.compile(optimizer=sgd, loss ='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x,train_y, batch_size=16, epochs=10000, verbose=1, validation_data=(val_x, val_y))
predict=model.predict(val_x)
df = pd.DataFrame(predict)
df.columns = ['Attacker','Midfielder','Defender','Goal Keeper']
df.index = val_data[:,0]
print(df)