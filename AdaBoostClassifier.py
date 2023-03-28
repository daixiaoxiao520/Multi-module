from msilib.schema import Binary
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib
import math
from sqlalchemy import true
matplotlib.use('TkAgg')

def GetSampleData(_filepath):
    iris = pd.read_excel(_filepath, header=None, engine='openpyxl')
    return iris.values
def GetSet(_data,random_state):
    sampleSet = _data[:, 2:]
    lableSet = _data[:,1]
    train_data, test_data, train_label, test_label = model_selection.train_test_split(sampleSet, lableSet, test_size=.3, random_state=random_state)
    clf = AdaBoostClassifier()
    clf.fit(train_data,train_label)
    predict = clf.predict(test_data)
    
    # outlable = pd.DataFrame(np.c_[test_label,predict])
    # writer = pd.ExcelWriter(r'D:\Tempwork\大论文数据\(GUS)_Label.xlsx',mode='a',engine='openpyxl')
    # print(outlable)
    # outlable.to_excel(writer,sheet_name='ABC')
    # writer.save()
    # writer.close()
    
    # print("Test_Label:\n",test_label)
    # print("Predict_Label:\n",predict)
    score = clf.decision_function(test_data)
    fpr, tpr, threshold = metrics.roc_curve(test_label, score)
    acc = metrics.accuracy_score(test_label, predict)
    auc = metrics.auc(fpr,tpr)
    # disp = ConfusionMatrixDisplay.from_estimator(clf,test_data, test_label,cmap=plt.cm.Blues,normalize='true')
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
    # return fpr,tpr,auc
    return auc

data = GetSampleData(r'D:\Tempwork\哈医大数据\498HZQM_H.xlsx')
a = np.empty(shape=[1,81],dtype=np.float64,order='C')
for i in range(0,81):
    print("第{}次采样".format(i+1))
    auc = GetSet(data,i)
    a[0,i] = auc
print(a)
print('AdaBoostClassifier多模态:')
print("平均值 = ",np.average(a,axis=1))
print("标准误差 = ",np.std(a,axis=1,ddof=0)/math.sqrt(81))
    
# data = GetSampleData(r'D:\Tempwork\哈医大数据\HZQM_HCT.xlsx')
# fpr,tpr,auc = GetSet(data,0)
# plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC Curves (AUC=%.3f)'%auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic Example')
# plt.legend(loc="lower right")
# plt.show()