import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import pickle
import sys

from transformers import AutoTokenizer

import argparse

parser = argparse.ArgumentParser(description='dataset generate')




def read_numbers(f):
    result = []
    while True:
        try:
            data = pickle.load(f)
            result += data
        except EOFError:
            break

    return result


roc_data = []
thresgroup = [0.6, 0.7, 0.8, 0.9]

for thres in thresgroup:
    ft = open(f'true_data_{thres}', 'rb')
    ff = open(f'false_data_{thres}', 'rb')

    data_positive = read_numbers(ft)
    data_negative = read_numbers(ff)

    print(len(data_positive))
    print(len(data_negative))

    data = np.concatenate([data_positive, data_negative]).reshape(-1, 1)
    true_labels = np.concatenate([np.ones(len(data_positive)), np.zeros(len(data_negative))])

    prob_class_1 = data.flatten()  # 使用标准化后的数据作为概率分数

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(true_labels, prob_class_1)
    roc_auc = auc(fpr, tpr)

    roc_data.append((fpr, tpr, roc_auc, thres))

    ft.close()
    ff.close()

plt.figure()
colors = ['darkorange', 'limegreen', 'dodgerblue', 'mediumvioletred']
# colors = ['darkorange', 'dodgerblue']
for i in range(len(roc_data)):
    # 绘制 ROC 曲线
    fpr, tpr, roc_auc, thres = roc_data[i]
    color = colors[i]
    plt.plot(fpr, tpr, color=color, lw=2, label='Threshold %0.2f: (area = %0.4f)' % (thres, roc_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Semantic ROC Curves for Different threshold')
plt.legend(loc="lower right")
plt.show()
