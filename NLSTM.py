'''Trains a Minimal RNN on the IMDB sentiment classification task.
The dataset is actually too small for Minimal RNN to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
'''
from __future__ import print_function

import os
import random

from keras_preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from nested_lstm import NestedLSTM
import pandas as pd
import get_data
from keras.callbacks import ReduceLROnPlateau
#设置学习速率
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
#固定随机种子，利于可重复性
def seed_tensorflow(see):
    os.environ['PYTHONHASHSEED'] = str(see)
    random.seed(see)
    np.random.seed(see)
    tf.random.set_seed(see)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_tensorflow(666)


max_features = 200
maxlen = 5 # cut texts after this number of words (among top max_features most common words)
batch_size = 128

# point=13242
# train='data/train.csv'
# valid='data/valid.csv'
# dftrain=pd.read_csv(train,low_memory=False)
#
# df_delay=dftrain['delay']
# df_delay=df_delay.iloc[1:point]
# pointv=3471
# dfvalid=pd.read_csv(valid,low_memory=False)
# dfv_delay=dfvalid['delay']
# dfv_delay=dfv_delay.iloc[1:pointv]
# print(dfv_delay)
# def fine_datas(data):
#     """
#     对每列都分别操作，一共操作4次
#     :param data:
#     :return:
#     """
#     names=['delay']
#
#
#
#
#     for name in names:
#             # 二维数组
#             data1=data
#             print(np.shape(data1))
#             print(len(data1))
#
#             for i in range(1,len(data1)):
#                 temp = 0
#                 if np.isnan(data1[i]):
#                     for k in range(5):
#                         if i-k-1>=0:
#
#                             temp += data1[i - k - 1]
#
#                     temp = (temp) / 5
#                     data1[i]= temp
#
#
#
#     return data1
#
# df_delay=fine_datas(df_delay)
# dfv_delay=fine_datas(dfv_delay)
#
# ss=MinMaxScaler()
# x_train=ss.fit_transform((df_delay.values).reshape(-1,1))
# x_test=ss.transform((dfv_delay.values).reshape(-1,1))
#
# def data_set(dataset,lookback):
#     #创建时间序列数据样本,lookback是回看步数
#   dataX,dataY=[],[]#初始化训练集和测试集的列表
#   for i in range(len(dataset)-lookback-1):
#         a=dataset[i:(i+lookback)]
#         dataX.append(a)
#         dataY.append(dataset[i+lookback])
#   return np.array(dataX),np.array(dataY)#转化为数组输出
# x_train1,y_train1=data_set(x_train,lookback=5)
# x_train=x_train1[:10593]
# y_train=y_train1[:10593]
# x_valid=x_train1[10593:]
# y_valid=y_train1[10593:]
# x_test,y_test=data_set(x_test,5)
# print(np.shape(y_train))
# x_tain=x_train[:,:,0]
# x_test=x_test[:,:,0]
trains,trainys,valids,validys,tests,testys,sss=get_data.get_data_nlstm()
names=['netstream_upward','netstream_downward']
#'delay','packet_loss',
historys=[]
scores_tests=[]

for name in names:
    #对每个指标都训练一个模型

    x_train=trains[names.index(name)+2]
    y_train=trainys[names.index(name)+2]
    x_valid=valids[names.index(name)+2]
    y_valid=valids[names.index(name)+2]
    x_test=tests[names.index(name)+2]
    y_test=testys[names.index(name)+2]
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 5))
    model.add(NestedLSTM(8, depth=3, dropout=0, recurrent_dropout=0))


    model.add(Dense(1, activation='linear'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='mape',
                  optimizer='adam' )

    model.summary()

    print('Train...')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=150,
                        validation_data=(x_valid, y_valid), callbacks=[reduce_lr]
                        )
    historys.append(history.history)
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    path=name+'_weights_nlstm.h5'
    model.save_weights(path)
    scores_tests.append(score)



np.save('historys_nlstm.npy',np.array(historys))
np.save('scores_tests_nlstm.npy',np.array(scores_tests))


