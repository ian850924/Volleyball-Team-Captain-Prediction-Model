# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


#==========資料前處理===========

df = pd.read_csv("data.csv")
ori_df = df.copy()
np_df = df.values


data = np.zeros((277,32))
data[:,5:7] = np_df[:,1:3]
data[:,9:-1] = np_df[:,4:-1]

df = pd.DataFrame(data,columns=['L', 'OS', 'S', 'O', 'MB', 'Years played', 'Height', 'M', 'F', 'Weight', 'Muscle',
       'Jumping', 'Endurance', 'catch serve', 'catch strong attack',
       'catch short trick', 'Underhand', 'Overhand', 'Block', 'Attack path',
       'Attack height', 'boa ball', 'Spike', 'Handle', 'Serve', 'Set',
       'Psychological quality', 'Leadership', 'Discipline', 'Perseverance',
       'Willingness', 'Y'])

for i in range(df.shape[0]):
    temp = ori_df['Position'][i]
    df[temp][i] = 1
    
    temp1 = ori_df['Sex'][i]
    df[temp1][i] = 1
    
    if ori_df['Y'][i] == "Yes":
        df['Y'][i] = 1
    else:
        df['Y'][i] = 0

x = df.drop(columns=['Y'])      
y = df['Y']


x['Years played'] = (x['Years played']-x['Years played'].min())/(x['Years played'].max()-x['Years played'].min())
x['Height'] = (x['Height']-x['Height'].min())/(x['Height'].max()-x['Height'].min())
x['Weight'] = (x['Weight']-x['Weight'].min())/(x['Weight'].max()-x['Weight'].min())
x.ix[:,10:] = (x.ix[:,10:]-x.ix[:,10:].min())/(x.ix[:,10:].max()-x.ix[:,10:].min())

x.to_csv("x.csv")
y.to_csv("y.csv")


#======讀前處理完的檔=========
'''
x = pd.read_csv("x.csv")
y = pd.read_csv("y.csv")
'''

break_n = 250
training_x = x[:][:break_n]
training_y = y[:][:break_n]
testing_x = x[:][break_n:]
testing_y = y[:][break_n:]

#from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

model = Sequential()
model.add(Dense(units=40, input_dim=31, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=30, kernel_initializer='uniform',activation='sigmoid'))
model.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x = training_x, y = training_y, validation_split=0.1, epochs=30, batch_size=30, verbose=2)
scores = model.evaluate(x = testing_x, y = testing_y)
print(scores[1])

#RNN

#modelRNN = Sequential()
#modelRNN.add(Embedding(output_dim=32,   #輸出的維度是32，希望將數字list轉換為32維度的向量
#     input_dim=31  #輸入的維度是3800，也就是我們之前建立的字典是3800字
#         )) #數字list截長補短後都是380個數字
#modelRNN.add(Dropout(0.7)) 	#隨機在神經網路中放棄70%的神經元，避免overfitting

#modelRNN.add(SimpleRNN(units=16))#建立16個神經元的RNN層
#modelRNN.add(Dense(units=256,activation='relu')) #建立256個神經元的隱藏層 #ReLU激活函數
#modelRNN.add(Dropout(0.7))
#modelRNN.add(Dense(units=1,activation='sigmoid'))#建立一個神經元的輸出層#Sigmoid激活函數

#modelRNN.summary()
#modelRNN.compile(loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']) #Loss function使用Cross entropy #adam最優化方法可以更快收斂

#train_history = modelRNN.fit(training_x,training_y, 
#         epochs=10, 
#         batch_size=1000,
#         verbose=2,
#         validation_split=0.2)


#scores = modelRNN.evaluate(testing_x, testing_y)
#print("Accuracy of RNN:",scores[1]) #使用test測試資料及評估準確率
prediction=model.predict(testing_x) 
print(prediction[:10])
print(testing_y[:10])  














