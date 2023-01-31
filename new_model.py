import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding, LSTM, Flatten, Dropout, Activation
from keras import datasets
from keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import sklearn
from nested_lstm import NestedLSTM
import get_data
from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error as mae
import get_data
#提取具有滑动窗口的数据
def get_right_res(ori,wei):
    res=[]
    k=np.array(ori).reshape(-1,1)
    for i in range(len(k)):
        temp=k[i][0]*wei
        res.append(temp)
    res=np.array(res).reshape(-1,1)
    return res


class new_model():
    def __init__(self):
        names=['delay','packet_loss','netstream_upward','netstream_downward']



        self.e1=0#mape
        self.e2=0#分别记录两个子模型的训练误差
        self.w1=0
        self.w2=0
        self.chars = ['delay', 'packet_loss', 'netstream_upward', 'netstream_downward']
        # xgboost参数集合
        self.xg_delay = XGBRegressor().set_params(n_estimators=200, learning_rate=0.05, gamma=0, colsample_bytree=0.6,
                                             subsample=0.6)
        self.xg_pl = XGBRegressor().set_params(n_estimators=150, learning_rate=0.1, gamma=0.2, colsample_bytree=0.6,
                                          subsample=0.7)
        self.xg_up = XGBRegressor().set_params(n_estimators=100, learning_rate=0.15, gamma=0.05, colsample_bytree=0.8,
                                          subsample=0.6)
        self.xg_down = XGBRegressor().set_params(n_estimators=100, learning_rate=0.1, gamma=0, colsample_bytree=0.6,
                                            subsample=0.8)
        self.xg_models = [self.xg_delay, self.xg_pl, self.xg_up, self.xg_down]
        # nlstm模型路径集合
        self.nlstms = ['delay_weights_nlstm.h5', 'packet_loss_weights_nlstm.h5', 'netstream_upward_weights_nlstm.h5',
                  'netstream_downward_weights_nlstm.h5']
        self.nlstm_models=self.get_nlstms(self.nlstms)
    def get_nlstms(self,paths):
        models=[]
        for path in paths:
            # 对每个指标都训练一个模型
            model = Sequential()
            model.add(Embedding(200, 5))
            if path=='netstream_downward_weights_nlstm.h5'or path=='netstream_upward_weights_nlstm.h5':
                model.add(NestedLSTM(8, depth=3, dropout=0, recurrent_dropout=0))
                model.add(Dense(1, activation='linear'))

                # try using different optimizers and different optimizer configs
                model.compile(loss='mape',
                              optimizer='adam'
                              )
                model.summary()
                model.load_weights(path)
                models.append(model)
            else:
                model.add(NestedLSTM(8, depth=2, dropout=0.4, recurrent_dropout=0.4))
                model.add(Dense(1, activation='linear'))

                # try using different optimizers and different optimizer configs
                model.compile(loss='mape',
                              optimizer='adam'
                              )
                model.summary()
                model.load_weights(path)
                models.append(model)
        return models
    def get_x_y(self,dataset,datasety,ind):
        x=dataset[ind]
        y=datasety[ind]
        return x,y
    def train(self,train_n,trainy_n,train_x,trainy_x):
        """
        1个数据集的所有特征一次训练完成

        :param train:
        :param trainy:
        :param chara:
        :return:
        """
        for i in range(4):

            model_xg = self.xg_models
            xg_model = self.xg_models[i]
            nlstm_model = self.nlstm_models[i]#加载了与训练结果
            xg_model.fit(train_x[:,:,0],trainy_x[:,i])
    def get_weight(self,row,colum):
        mapes_xg = np.load('mapes_xgboost.npy').tolist()
        mapes_nlstm=np.load('mapes_nlstm.npy').tolist()

        mape_xg=(mapes_xg[row])[colum]
        mape_nlstm=(mapes_nlstm[row])[colum]
        w_xg=mape_nlstm/(mape_xg+mape_nlstm)
        w_n=mape_xg/(mape_nlstm+mape_xg)
        return w_xg,w_n
    def get_ss(self,sss,row):
        ss=sss[row]
        return ss



    def predict(self,indic):
        """
        用数据进行验证得值,一个数据集所有特征的预测
        :param valid:
        :param validy:
        :return:
        """
        #一次把4个指标全预测
        #xgboost的训练结果
        indcs=['train','valid','test']
        pres_res=[]

        for name in self.chars:
            ind=self.chars.index(name)#特征

            indx=indcs.index(indic)#训，测，验
            print(ind)
            print(indx)
            w_x,w_n=self.get_weight(ind,indx)
            print(w_x)
            print(w_n)
            pre_nlstm_train=np.load('nlstm_pre_train.npy').tolist()
            pre_nlstm_valid=np.load('nlstm_pre_valid.npy').tolist()
            pre_nlstm_test=np.load('nlstm_pre_test.npy').tolist()
            nlstm_pre=[pre_nlstm_train,pre_nlstm_valid,pre_nlstm_test]
            pre_xg_train=np.load('xg_pre_train.npy').tolist()
            pre_xg_valid=np.load('xg_pre_valid.npy').tolist()
            pre_xg_test=np.load('xg_pre_test.npy').tolist()
            xg_pre=[pre_xg_train,pre_xg_valid,pre_xg_test]
            temp=(np.array(xg_pre[indx]))[:,ind]

            pre_x=get_right_res(temp,w_x)
            #xgboost部分的结果
            temp=(nlstm_pre[indx])[ind]
            pre_n =get_right_res(temp, w_n)
            #nlstm部分的结果

            pre_model=pre_n+pre_x
            # print(pre_model)
            pres_res.append(pre_model)


            #新方法的结果
        return pres_res

trains,trainys,valids,validys,tests,testys,sss=get_data.get_data_nlstm()
trainx,trainyx,validx,validyx,testx,testyx,ss=get_data.train_valid()
new=new_model()
# new.train(trains,trainys,trainx,trainyx)
res_train=new.predict('train')
res_valid=new.predict('valid')
res_test=new.predict('test')


trains=np.array(res_train)
np.save('pre_train__new_model.npy',trains)

trains=np.array(res_valid)
np.save('pre_valid__new_model.npy',trains)

trains=np.array(res_test)
np.save('pre_test__new_model.npy',trains)



