import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
lookback=5
#取数据
point=13242
train='data/train.csv'
valid='data/valid.csv'
dftrain=pd.read_csv(train,low_memory=False)

df_delay=dftrain[['delay','packet_loss','netstream_upward','netstream_downward']]
df_delay=df_delay.iloc[1:point]
pointv=3471
dfvalid=pd.read_csv(valid,low_memory=False)
dfv_delay=dfvalid[['delay','packet_loss','netstream_upward','netstream_downward']]
dfv_delay=dfv_delay.iloc[1:pointv]
print(dfv_delay)
#缺失值填充
def fine_data(data):
    """
    对每列都分别操作，一共操作4次
    :param data:
    :return:
    """
    names=['delay','packet_loss','netstream_upward','netstream_downward']




    for name in names:
            # 二维数组
            data1 = data.loc[:, [name]].values

            for i in range(np.shape(data1)[0]):
                temp = 0
                if np.isnan(data1[i][0]):
                    for k in range(lookback):
                        if i-k-1>=0:

                            temp += data1[i - k - 1][0]

                    temp = (temp) / lookback
                    data1[i][0] = temp

            data.loc[:, [name]] = data1

    return data

df_delay=fine_data(df_delay)
dfv_delay=fine_data(dfv_delay)
df_delay.to_csv("temp.csv")
dfv_delay.to_csv("tempv.csv")

#标准化(-1,1)
ss=StandardScaler()
df_delay=ss.fit_transform((df_delay.values).reshape(-1,4))
dfv_delay=ss.transform((dfv_delay.values).reshape(-1,4))


#滑动窗口变换数据
def data_set(dataset,lookback):
    #创建时间序列数据样本,lookback是回看步数
  dataX,dataY=[],[]#初始化训练集和测试集的列表
  for i in range(len(dataset)-lookback-1):
        a=dataset[i:(i+lookback)]
        dataX.append(a)
        dataY.append(dataset[i+lookback])
  return np.array(dataX),np.array(dataY)#转化为数组输出

#返回训练集和验证集
def train_valid():
    delay_train,delay_trainy=data_set(df_delay,lookback)
    delayv_train, delayv_trainy = data_set(dfv_delay, lookback)
    x_train = delay_train[:10593]
    y_train =delay_trainy[:10593]
    x_valid = delay_train[10593:]
    y_valid = delay_trainy[10593:]
    x_test=delayv_train
    y_test=delayv_trainy

    return x_train,y_train,x_valid,y_valid,x_test,y_test,ss

def fine_datas(data):
    """
    对每列都分别操作，一共操作4次
    :param data:
    :return:
    """
    names=['delay']




    for name in names:
            # 二维数组
            data1=data
            print(np.shape(data1))
            print(len(data1))

            for i in range(1,len(data1)):
                temp = 0
                if np.isnan(data1[i]):
                    for k in range(5):
                        if i-k-1>0:

                            temp += data1[i - k - 1]

                    temp = (temp) / 5
                    data1[i]= temp



    return data1

def get_data_nlstm():
    """
    找到4个指标对应的训练，验证和测试集合
    :return:
    """

    trains=[]
    trainys=[]
    valids=[]
    validys=[]
    tests=[]
    testys=[]
    names=['delay','packet_loss','netstream_upward','netstream_downward']
    sss=[]
    for name in names:
        point = 13242
        train = 'data/train.csv'
        valid = 'data/valid.csv'
        dftrain = pd.read_csv(train, low_memory=False)

        df_delay = dftrain[name]
        df_delay = df_delay.iloc[1:point]
        pointv = 3471
        dfvalid = pd.read_csv(valid, low_memory=False)
        dfv_delay = dfvalid[name]
        dfv_delay = dfv_delay.iloc[1:pointv]
        df_delay = fine_datas(df_delay)
        dfv_delay = fine_datas(dfv_delay)
        ss = StandardScaler()
        sss.append(ss)
        x_train = ss.fit_transform((df_delay.values).reshape(-1, 1))
        x_test = ss.transform((dfv_delay.values).reshape(-1, 1))
        x_train1, y_train1 = data_set(x_train, lookback=5)
        x_train = x_train1[:10593]
        y_train = y_train1[:10593]
        x_valid = x_train1[10593:]
        y_valid = y_train1[10593:]
        x_test, y_test = data_set(x_test, 5)
        print(np.shape(y_train))
        x_train = x_train[:, :, 0]
        x_valid=x_valid[:,:,0]
        x_test = x_test[:, :, 0]
        trains.append(x_train)
        trainys.append(y_train)
        valids.append(x_valid)
        validys.append(y_valid)
        tests.append(x_test)
        testys.append(y_test)
    return trains,trainys,valids,validys,tests,testys,sss




