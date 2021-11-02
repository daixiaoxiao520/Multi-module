def result(TP,TN,FN,FP):
    pre = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = (2*pre*recall)/(pre+recall)
    npv = TN/(TN+FN)
    ppv = TP/(TP+FP)
    print('F1score:',F1,' Recall:',recall,' PPV:',ppv, ' NPV:',npv)

result(0.85,0.54,0.15,0.46)