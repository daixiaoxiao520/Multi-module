# -*- coding: utf-8 -*-

from csv import writer
from tokenize import String
from markupsafe import string
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import f1_score


def GetSampleData(_filepath):
    iris = pd.read_excel(_filepath, header=None, engine='openpyxl')
    return iris.values
def GetSet(_data,random_state):
    from sklearn import model_selection,svm
    sampleSet = _data[:, 2:]
    lableSet = _data[:,1]
    train_data, test_data, train_label, test_label = model_selection.train_test_split(sampleSet, lableSet, test_size=.3, random_state=random_state)
    classfier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    test_predict_score = classfier.fit(train_data, train_label).decision_function(test_data)
    predict =  classfier.fit(train_data, train_label).predict(test_data)
    
    # outlable = pd.DataFrame(np.c_[test_label,predict])
    # writer = pd.ExcelWriter(r'D:\Tempwork\大论文数据\(GUS)_Label.xlsx',mode='a',engine='openpyxl')
    # print(outlable)
    # outlable.to_excel(writer,sheet_name='SVM')
    # writer.save()
    # writer.close()
    
    # print("Test_Label:\n",test_label)
    # print("Predict_Label:\n",predict)
    fpr, tpr, threshold = sk.metrics.roc_curve(test_label, test_predict_score)
    acc = sk.metrics.accuracy_score(test_label,predict)
    auc = sk.metrics.auc(fpr,tpr)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # disp = ConfusionMatrixDisplay.from_estimator(classfier.fit(train_data, train_label),test_data, test_label, cmap=plt.cm.Blues, normalize='true')
    cm = confusion_matrix(test_label,predict,normalize='true')
    print("敏感性(Sensitive) =",cm[1,1])
    print("特异性(Specificity) =",cm[0,0])
    print("阳性预测值 =",cm[1,1]/(cm[0,1]+cm[1,1]))
    print("阴性预测值 =",cm[0,0]/(cm[0,0]+cm[1,0]))
    print("F1-score =",f1_score(test_label, predict, average="binary"))
    print("Recall =",cm[1,1])
    print("精准度(Precision)",cm[1,1]/(cm[0,1]+cm[1,1]))
    print("ACC =",acc)
    print("AUC =",auc)
    print("-"*60)
    # return fpr,tpr,acc,auc
    return auc


data = GetSampleData(r'D:\Tempwork\哈医大数据\498HZQM_H.xlsx')
a = np.empty(shape=[1,81],dtype=np.float64,order='C')
for i in range(0,81):
    print("第{}次采样".format(i))
    auc = GetSet(data,i)
    a[0,i] = auc
print(a)
print('498-SVM单模态:')
print("平均值 = ",np.average(a,axis=1))
print("标准误差 = ",np.std(a,axis=1,ddof=1)/math.sqrt(81))



# data = GetSampleData(r'D:\Tempwork\哈医大数据\HZQM_HCT.xlsx')
# fpr,tpr,acc,auc= GetSet(data,1)
# plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC Curves (AUC=%.3f)'%auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('Specificity')
# plt.ylabel('Sensitivity')
# plt.title('Receiver Operating Characteristic Example')
# plt.legend(loc="lower right")
# plt.show()