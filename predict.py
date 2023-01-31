from __future__ import print_function
from xgboost import XGBRegressor
import get_data
from matplotlib import pyplot as plt
import numpy as np
import os
from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape


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
def predict_visiual(truth,pre,dur):
    #.reshape(-1,1)

    plt.plot(truth[:dur],color='b',label='ture value')
    plt.plot(pre[:dur],color='r',label='pre value')
    plt.legend()

    plt.show()


#XGBoost预测
# train,trainy,valid,validy,test,testy,ss=get_data.train_valid()
# train = train[:,:,0]
# valid=valid[:,:,0]
# test=test[:,:,0]
#
#
#
# #delay
# model_delay=XGBRegressor()
# model_delay.set_params(n_estimators=200,learning_rate=0.05,gamma=0,colsample_bytree=0.6,subsample=0.6)
# trainy_temp=trainy[:,0]
# model_delay.fit(train,trainy_temp)
# res_train_delay=model_delay.predict(train).reshape(-1,1)
# res_valid_delay=model_delay.predict(valid).reshape(-1,1)
# res_test_delay=model_delay.predict(test).reshape(-1,1)
#
#
# model_pl=XGBRegressor()
# model_pl.set_params(n_estimators=150,learning_rate=0.1,gamma=0.2,colsample_bytree=0.6,subsample=0.7)
# trainy_temp=trainy[:,1]
# validy_temp=validy[:,1]
# testy_temp=testy[:,1]
# dur=50
# model_pl.fit(train,trainy_temp)
# res_train_pl=model_pl.predict(train).reshape(-1,1)
# res_valid_pl=model_pl.predict(valid).reshape(-1,1)
# res_test_pl=model_pl.predict(test).reshape(-1,1)
#
#
# model=XGBRegressor()
# model.set_params(n_estimators=100,learning_rate=0.15,gamma=0.05,colsample_bytree=0.8,subsample=0.6)
# trainy_temp=trainy[:,2]
#
# model.fit(train,trainy_temp)
# res_train_up=model.predict(train).reshape(-1,1)
# res_valid_up=model.predict(valid).reshape(-1,1)
# res_test_up=model.predict(test).reshape(-1,1)
#
#
# model=XGBRegressor()
# model.set_params(n_estimators=100,learning_rate=0.1,gamma=0,colsample_bytree=0.6,subsample=0.8)
# trainy_temp=trainy[:,3]
# model.fit(train,trainy_temp)
# res_train_down=model.predict(train).reshape(-1,1)
# res_valid_down=model.predict(valid).reshape(-1,1)
# res_test_down=model.predict(test).reshape(-1,1)
#
# res_train=np.hstack((res_train_delay,res_train_pl,res_train_up,res_train_down))
# res_valid=np.hstack((res_valid_delay,res_valid_pl,res_valid_up,res_valid_down))
# res_test=np.hstack((res_test_delay,res_test_pl,res_test_up,res_test_down))
#
#
#
# #用于绘制在各数据集上的预测情况，使用时修改res_test为res_train,testy为trainy就可以画在训练集上的表现，验证集上的同理
# dur=50
# for i in range(4):
#     #每个特征的train集情况部分
#     pre_tra=res_test[:,i]
#     true_tre=testy[:,i]
#     predict_visiual(true_tre,pre_tra,dur)
#
# #计算mape
#
#
# np.save('xg_pre_train.npy',np.array(res_train))
# np.save('xg_pre_valid.npy',np.array(res_valid))
# np.save('xg_pre_test.npy',np.array(res_test))
# trains=[]
# valids=[]
# tests=[]
# for i in range(4):
#
#     trainy_temp=trainy[:,i].reshape(-1,1)
#     validy_temp=validy[:,i].reshape(-1,1)
#     testy_temp=testy[:,i].reshape(-1,1)
#     print(res_train[:,i])
#     trains.append(mape(trainy_temp,res_train[:,i]))
#     valids.append(mape(validy_temp,res_valid[:,i]))
#     tests.append(mape(testy_temp,res_test[:,i]))
#
#
# trains=np.array(trains)
# np.save('train_mape_xgboost.npy',trains)
#
# valids=np.array(valids)
# np.save('valid_mape_xgboost.npy',valids)
#
# tests=np.array(tests)
# np.save('test_mape_xgboost.npy',tests)



paths=['delay_weights_nlstm.h5','packet_loss_weights_nlstm.h5','netstream_upward_weights_nlstm.h5','netstream_downward_weights_nlstm.h5']
trains,trainys,valids,validys,tests,testys,sss=get_data.get_data_nlstm()
names=['delay','packet_loss','netstream_upward','netstream_downward']
#
scores_train=[]
scores_valid=[]
scores_test=[]
pre_trains=[]
pre_valids=[]
pre_tests=[]
dur=100
for name in names:
    #对每个指标都训练一个模型

    x_train=trains[names.index(name)]
    y_train=trainys[names.index(name)]
    x_valid=valids[names.index(name)]
    y_valid=validys[names.index(name)]
    x_test=tests[names.index(name)]
    y_test=testys[names.index(name)]
    path=paths[names.index(name)]
    ss=sss[names.index(name)]
    if names.index(name)<2:
        model = Sequential()
        model.add(Embedding(200, 5))
        model.add(NestedLSTM(8, depth=2, dropout=0.8, recurrent_dropout=0.8))
        model.add(Dense(1, activation='linear'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='mape',
                      optimizer='adam'
                      )
        model.summary()
        model.load_weights(path)
    else:
        model = Sequential()
        model.add(Embedding(200, 5))
        model.add(NestedLSTM(8, depth=3, dropout=0, recurrent_dropout=0))
        model.add(Dense(1, activation='linear'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='mape',
                      optimizer='adam'
                      )
        model.summary()
        model.load_weights(path)


    res_train=model.predict(x_train)
    res_valid=model.predict(x_valid)
    res_test=model.predict(x_test)
    pre_trains.append(res_train)
    pre_valids.append(res_valid)
    pre_tests.append(res_test)
    mape_train=mape(y_train,res_train)
    scores_train.append(mape_train)
    mape_valid=mape(y_valid, res_valid)
    scores_valid.append(mape_valid)
    mape_test=mape(y_test, res_test)
    scores_test.append(mape_test)

    predict_visiual(y_train, res_train,dur)
    predict_visiual(y_valid, res_valid,dur)
    predict_visiual(y_test, res_test,dur)

trains=np.array(scores_train)
np.save('train_mape_nlstm.npy',trains)

valids=np.array(scores_valid)
np.save('valid_mape_nlstm.npy',valids)

tests=np.array(scores_test)
np.save('test_mape_nlstm.npy',tests)

np.save('nlstm_pre_train.npy',np.array(pre_trains))
np.save('nlstm_pre_valid.npy',np.array(pre_valids))
np.save('nlstm_pre_test.npy',np.array(pre_tests))



# pre_train=np.load('pre_train__new_model.npy').tolist()
# pre_valid=np.load('pre_valid__new_model.npy').tolist()
# pre_test=np.load('pre_test__new_model.npy').tolist()
# trains,trainys,valids,validys,tests,testys,sss=get_data.get_data_nlstm()
# dur=50
# ress=[]
#
# for i in range(4):
#     res=[]
#     p_train=pre_train[i]
#     p_train=sss[i].transform(p_train)
#     p_valid=pre_valid[i]
#     p_valid=sss[i].transform(p_valid)
#     p_test=pre_test[i]
#     p_test=sss[i].transform(p_test)
#     # t_train=sss[i].inverse_transform(trainys[i])
#     # t_valid=sss[i].inverse_transform(validys[i])
#     # t_test=sss[i].inverse_transform(testys[i])
#     t_train=trainys[i]
#     t_valid=validys[i]
#     t_test=testys[i]
#     mape_train=mape(t_train,p_train)
#     mape_valid=mape(t_valid,p_valid)
#     mape_test=mape(t_test,p_test)
#     res=[mape_train,mape_valid,mape_test]
#     ress.append(res)
#     predict_visiual(sss[i].inverse_transform(t_train),sss[i].inverse_transform(p_train),dur)
#     predict_visiual(sss[i].inverse_transform(t_valid),sss[i].inverse_transform(p_valid), dur)
#     predict_visiual(sss[i].inverse_transform(t_test),sss[i].inverse_transform(p_test), dur)
# print(ress)

