# -*- coding: utf-8 -*-

import matplotlib
import sklearn; matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc, accuracy_score   ###计算roc和auc
# from sklearn import model_selection
# from sklearn import preprocessing
# from sklearn.metrics import ConfusionMatrixDisplay
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# print(type(X))
# print(type(y))
# ##变为2分类
# X, y = X[y != 2], y[y != 2]
# Import some data to play with
# iris = pd.read_excel(r'D:\TempDataSet\workSource\training_data_feature\HZQM_H.xlsx', header=None, engine='openpyxl')
# data = iris.values
# min_max_scaler = preprocessing.MinMaxScaler()

# X = data[:, 2::]  # 得到样本集
# y = data[:, 1]  # 得到标签集

##变为2分类
# X, y = X[y != 0], y[y != 0]  # 通过取y不等于1来取两种类别
# Add noisy features to make the problem harder添加扰动
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets划分样本集
# train_data, test_data, train_label, test_label = model_selection.train_test_split(X, y, test_size=.3, random_state=0)
# train_data用于训练的样本集, test_data用于测试的样本集, train_label训练样本对应的标签集, test_label测试样本对应的标签集

# Learn to predict each class against the other分类器设置
# classifier = svm.SVC(kernel='linear', C=0.01).fit(train_data, train_label)

# svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)  # 使用核函数为线性核，参数默认，创建分类器
###通过decision_function()计算得到的test_predict_label的值，用在roc_curve()函数中
# test_predict_label = svm.fit(train_data, train_label).decision_function(test_data)
# predict =  svm.fit(train_data, train_label).predict(test_data)
# 首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集
# Compute ROC curve and ROC area for each class#计算tp,fp
# 通过测试样本输入的标签集和模型预测的标签集进行比对，得到fp,tp,不同的fp,tp是算法通过一定的规则改变阈值获得的
# fpr, tpr, threshold = roc_curve(test_label, test_predict_label)  ###计算真正率和假正率
# acc = accuracy_score(test_label,predict)

# roc_auc = auc(fpr, tpr)  ###计算auc的值，auc就是曲线包围的面积，越大越好
# print("AUC:",roc_auc)
# print("ACC:",acc)
# disp = ConfusionMatrixDisplay.from_estimator(svm.fit(train_data, train_label),test_data, test_label,cmap=plt.cm.Blues,display_labels=None,normalize='true')
# print("Confusion_Matrix",disp.confusion_matrix)
# lw = 2
# plt.figure()

def GetSampleData(_filepath):
    iris = pd.read_excel(_filepath, header=None, engine='openpyxl')
    return iris.values
def GetSet(_data):
    from sklearn import model_selection,svm
    sampleSet = _data[:, 2::]
    lableSet = _data[:,1]
    random_state = np.random.RandomState(0)
    n_samples, n_features = sampleSet.shape
    sampleSet = np.c_[sampleSet, random_state.randn(n_samples, 200 * n_features)]
    train_data, test_data, train_label, test_label = model_selection.train_test_split(sampleSet, lableSet, test_size=.3, random_state=0)
    classfier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    test_predict_label = classfier.fit(train_data, train_label).decision_function(test_data)
    predict =  classfier.fit(train_data, train_label).predict(test_data)
    fpr, tpr, threshold = sk.metrics.roc_curve(test_label, test_predict_label)
    acc = sk.metrics.accuracy_score(test_label,predict)
    return fpr,tpr,acc


data1 = GetSampleData(r'D:\TempDataSet\workSource\training_data_feature\HQM_H.xlsx')
data2 = GetSampleData(r'D:\TempDataSet\workSource\training_data_feature\HQM_C.xlsx')
data3 = GetSampleData(r'D:\TempDataSet\workSource\training_data_feature\HQM_T.xlsx')
data4 = GetSampleData(r'D:\TempDataSet\workSource\training_data_feature\HQM_HC.xlsx')
data5 = GetSampleData(r'D:\TempDataSet\workSource\training_data_feature\HQM_HT.xlsx')
data6 = GetSampleData(r'D:\TempDataSet\workSource\training_data_feature\HQM_CT.xlsx')
data7 = GetSampleData(r'D:\TempDataSet\workSource\training_data_feature\HQM_HCT.xlsx')
data8 = GetSampleData(r'D:\TempDataSet\workSource\training_data_feature\HQM_HCTL.xlsx')
fpr1,tpr1,acc1 = GetSet(data1)
fpr2,tpr2,acc2 = GetSet(data2)
fpr3,tpr3,acc3 = GetSet(data3)
fpr4,tpr4,acc4 = GetSet(data4)
fpr5,tpr5,acc5 = GetSet(data5)
fpr6,tpr6,acc6 = GetSet(data6)
fpr7,tpr7,acc7 = GetSet(data7)
fpr8,tpr8,acc8 = GetSet(data8)
plt.plot(fpr1, tpr1, color='darkorange',lw=2, label='GUS(AUC = 0.77)')
plt.plot(fpr2, tpr2, color='red',lw=2, label='CUS(AUC = 0.76)')
plt.plot(fpr3, tpr3, color='forestgreen',lw=2, label='EUS(AUC = 0.76)')
plt.plot(fpr4, tpr4, color='lightblue',lw=2, label='(G+C)US(AUC = 0.76)')
plt.plot(fpr5, tpr5, color='darksalmon',lw=2, label='(G+E)US(AUC = 0.7)')
plt.plot(fpr6, tpr6, color='darkseagreen',lw=2, label='(C+E)US(AUC = 0.73)')
plt.plot(fpr7, tpr7, color='deeppink',lw=2, label='(G+C+E)US(AUC = 0.76)')
plt.plot(fpr8, tpr8, color='darkviolet',lw=2, label='(G+C+E)US+Clinical(AUC = 0.74)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
    