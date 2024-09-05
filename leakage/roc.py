import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import pickle
import sys

from transformers import AutoTokenizer

prompt = (
    "You are a math tutor who helps students of all levels understand and solve mathematical problems. "
    "Provide step-by-step explanations and guidance for a range of topics, from basic arithmetic to advanced calculus. "
    "Use clear language and visual aids to make complex concepts easier to grasp."
)


def read_numbers(filename):
    with open(filename, 'r') as file:
        numbers = np.array([float(line.strip()) for line in file], dtype=np.float64)
    return numbers

def calculate_and_print_mean(filename):
    data = read_numbers(filename)
    mean_value = np.mean(data)
    print(f"{mean_value}")
    return mean_value

roc_data = []

id = [1,2,4,8]
for word in id:

    miss = f'/mnt/data1/pzx/sglang/lab/data/roc/miss.txt'
    hit = f'/mnt/data1/pzx/sglang/lab/data/roc/hit.txt'

    # mean_1 = calculate_and_print_mean(miss)
    # mean_1_hit = calculate_and_print_mean(hit)

    data_positive = read_numbers(miss)
    data_negative = read_numbers(hit)

    print(len(data_positive))
    print(len(data_negative))
    data = np.concatenate([data_positive, data_negative]).reshape(-1, 1)
    true_labels = np.concatenate([np.ones(len(data_positive)), np.zeros(len(data_negative))])

    prob_class_1 = data.flatten()

    fpr, tpr, thresholds = roc_curve(true_labels, prob_class_1)
    roc_auc = auc(fpr, tpr)

    roc_data.append((fpr, tpr, roc_auc, word))

with open(f'data/roc.pkl', 'wb') as file:
        pickle.dump(roc_data, file)
