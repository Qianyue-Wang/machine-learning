import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding, LSTM, Flatten, Dropout, Activation
from keras import datasets
from keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import get_data
import os
import random
from keras.callbacks import ReduceLROnPlateau
#设置学习速率
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
#提取具有滑动窗口的数据
train,trainy,valid,validy,test,testy,ss=get_data.train_valid()
def seed_tensorflow(see):
    os.environ['PYTHONHASHSEED'] = str(see)
    random.seed(see)
    np.random.seed(see)
    tf.random.set_seed(see)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_tensorflow(666)


tf.random.set_seed(666)
np.random.seed(666)
model = Sequential()
model.add(LSTM(units=8, activation='relu',
                   input_shape=(train.shape[1], train.shape[2]),
               return_sequences=True))
model.add(Dropout(0.8))
model.add(LSTM(units=8, activation='relu', return_sequences=True))
model.add(Dropout(0.8))
model.add(LSTM(units=8, return_sequences=True))
model.add(Dropout(0.8))
model.add(LSTM(units=8))
model.add(Dropout(0.8))
model.add(Dense(trainy.shape[1], activation='linear'))
model.compile(loss='mape', optimizer='adam')



# nb_lstm_outputs = 12  #神经元个数
# nb_time_steps = 5  #时间序列长度
# nb_input_vector = 1 #输入序列
# model = Sequential()
# model.add(LSTM(50, activation='relu',input_shape=(train.shape[1], train.shape[2]), return_sequences=True))
# model.add(Flatten())
#
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.summary()
history=model.fit(train, trainy, epochs=150, batch_size=128, validation_data=(valid, validy), verbose=1, shuffle=False, callbacks=[reduce_lr])
np.save('history_stacked_lstm.npy',history.history)
score = model.evaluate(test, testy,batch_size=128, verbose=1)
model.save_weights('stacked_lstm.h5')
scores=[]
scores.append(score)
np.save('score_stacked_lstm.npy',scores)
# print(score)