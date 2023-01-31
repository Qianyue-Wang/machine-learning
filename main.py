import matplotlib.pyplot as plt
import numpy as np
#仅对某个link 和 delay作预测
#图像规律感受
# 13243
# point=13242
# point=50#从图像看出具有明显周期性
# import pandas as pd
# from matplotlib import pyplot as plt
# train='data/train.csv'
# dftrain=pd.read_csv(train)
# names=['delay','packet_loss','netstream_upward','netstream_downward']
# print()
# for name in names:
#     df_delay = dftrain[name]
#     df_delay = df_delay.iloc[1:point]
#     df_index = dftrain['time']
#     df_index = df_index[1:point]
#     df_delay.index = df_index.values
#     print(df_delay)
#     ma=max(df_delay.values)
#     plt.plot( df_delay.values)
#     plt.ylabel('value')
#     plt.ylim(0,ma+0.5)
#     plt.show()


#对训练过程绘制学习曲线可视化
#nlstm
# nlstm_history=np.load('historys_nlstm.npy',allow_pickle=True)
# names=['delay','packet_loss','netstream_upward','netstream_downward']
#
# for i in range(4):
#     name=names[i]
#     fig=plt.figure()
#     history=nlstm_history[i]
#     loss=history['loss']
#     val_loss=history['val_loss']
#     plt.plot(loss,color='b',label='train set')
#     plt.plot(val_loss,color='r',label='valid set')
#     plt.legend()
#     plt.xlabel('iteration')
#     plt.ylabel('loss value')
#     plt.show()



#绘制拟合图
#统计个模型上的mape值
mape_train=np.load('train_mape_nlstm.npy').tolist()
mape_valid=np.load('valid_mape_nlstm.npy').tolist()
mape_test=np.load('test_mape_nlstm.npy').tolist()
print(mape_train)
print(mape_valid)
print(mape_test)


mape_train=np.load('train_mape_xgboost.npy').tolist()
mape_valid=np.load('valid_mape_xgboost.npy').tolist()
mape_test=np.load('test_mape_xgboost.npy').tolist()
print(mape_train)
print(mape_valid)
print(mape_test)
#提出方法示意图（亿图）

#制作xgboost和nlstms的mape的记录:
# mape_nlstms=[[24.9,41.7,27.7],[3.3,0.9,2.85],[99.7,99.8,100],[99.6,99.1,100]]
# valids=np.array(mape_nlstms)
# np.save('mapes_nlstm.npy',valids)
# mape_xgs=[]
# mape_train=np.load('train_mape_xgboost.npy').tolist()
# mape_valid=np.load('valid_mape_xgboost.npy').tolist()
# mape_test=np.load('test_mape_xgboost.npy').tolist()
# mape_xgs.append(mape_train)
# mape_xgs.append(mape_valid)
# mape_xgs.append(mape_test)
#
# valids=np.array(mape_xgs)
# print(valids)
# valids=valids.T
# print(valids)
# np.save('mapes_xgboost.npy',valids)
#
# data=np.load('mapes_xgboost.npy').tolist()
# print(data[1])


