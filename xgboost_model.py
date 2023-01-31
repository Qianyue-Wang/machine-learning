import copy

import sklearn

import get_data
from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error as mape


train,trainy,valid,validy,test,testy,ss=get_data.train_valid()

train = train[:,:,0]
valid=valid[:,:,0]
test=test[:,:,0]

#初始状况
model_delay=XGBRegressor()
model_pack_loss=XGBRegressor()
model_down=XGBRegressor()
model_up=XGBRegressor()
models=[model_delay,model_pack_loss,model_down,model_up]
scores=[0,0,0,0]
for i in range(4):
    print(i)
    trainy_temp=trainy[:,i]
    validy_temp=validy[:,i]
    model_temp=models[i]
    score=scores[i]
    model_temp.fit(train,trainy_temp)
    pret=model_temp.predict(train)
    pre=model_temp.predict(valid)
    res=mape(validy_temp,pre)
    resy=mape(trainy_temp,pret)
    print(res)
    print(resy)
#参数调优
for i in range(0,4):
    print(i)
    mape_train=5000000000000
    mape_valid=100
    mape_test=100

    trainy_temp=trainy[:,i]
    validy_temp=validy[:,i]
    testy_temp=testy[:,i]
    model_temp=models[i]

    n_estimators = [i for i in range(50, 500, 50)]
    gamma = [0, 0.05, 0.1, 0.15, 0.2]
    learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2]
    colsample_bytree = [0.6, 0.7, 0.8, 0.9]
    subsample = [ 0.6, 0.7, 0.8, 0.9]

    #模型调参部分
    # for ne in n_estimators:
    #     for ga in gamma:
    #         for lr in learning_rate:
    #             for cb in colsample_bytree:
    #                 for sub in subsample:
    #                     model_tt=copy.copy(model_temp)
    #                     model_tt.set_params(n_estimators=ne,learning_rate=lr,gamma=ga,colsample_bytree=cb,subsample=sub)
    #                     model_tt.fit(train,trainy_temp)
    #                     pre_train=model_tt.predict(train)
    #                     pre_valid=model_tt.predict(valid)
    #                     pre_test=model_tt.predict(test)
    #                     res_train=mape(trainy_temp,pre_train)
    #                     res_valid=mape(validy_temp,pre_valid)
    #                     res_test=mape(testy_temp,pre_test)
    #                     print(res_train)
    #                     print(res_valid)
    #                     print(res_test)
    #
    #
    #                     if res_train<mape_train and res_valid<mape_valid :
    #                         mape_train=res_train
    #                         mape_valid=res_valid
    #                         mape_test=res_test
    #                         print(res_train)
    #                         print(res_valid)
    #                         print(res_test)
    #                         print(ne)
    #                         print(ga)
    #                         print(lr)
    #                         print(cb)
    #                         print(sub)
    #                         print()
    #模型预测部分




