import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

inputfile = r"G:\feature_result\498TRAIN_DATA\C_Z_FEATURE\f4_maxpool_feature.csv"  #输入数据文件
labelfile = r'G:\feature_result\498TRAIN_DATA\label.xlsx'
data = pd.read_csv(inputfile,header=None)
print("load_data end!")
print(data)
label = pd.read_excel(labelfile,header=None)
print("load_label end!")
print(label)
lasso = Lasso(alpha=0.01,max_iter=100000)   #调用lasso，设置λ值为1000
lasso.fit(data,label)
print("相关系数为：",np.round(lasso.coef_,5))

#计算相关系数非0的个数
print("相关系数非0个数为：",np.sum(lasso.coef_ != 0))

mask = lasso.coef_ != 0
print("相关系数是否为0：",mask)
mask = list(mask)
data_write = data.iloc[:,mask]
print(data_write)
print(data_write.shape)
writer = pd.ExcelWriter(r"G:\feature_result\498TRAIN_DATA\C_Z_FEATURE\lasso\f4_lasso_feature_num=" +str( data_write.shape[1])+".xlsx")
data_write.to_excel(writer,"page_1",float_format='%f')
writer.save()
writer.close()