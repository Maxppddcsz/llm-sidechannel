import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sys

def load_roc_data(filepaths):
    combined_roc_data = {}
    for filepath in filepaths:
        with open(filepath, 'rb') as file:
            roc_data = pickle.load(file)
            for entry in roc_data:
                key = entry[3]  # word 数量
                if key not in combined_roc_data:
                    combined_roc_data[key] = []
                combined_roc_data[key].append(entry)
    return combined_roc_data

def average_roc(roc_data):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_list = []
    for fpr, tpr, roc_auc, word in roc_data:
        mean_tpr_list.append(np.interp(mean_fpr, fpr, tpr))
    
    mean_tpr = np.mean(mean_tpr_list, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    
    return mean_fpr, mean_tpr, mean_auc

if __name__ == "__main__":

    filepaths = ["data/roc.pkl"]

    combined_roc_data = load_roc_data(filepaths)

    plt.figure(figsize=(16, 10))

    colors = plt.cm.get_cmap('tab20', len(combined_roc_data))
    for idx, (word, roc_data) in enumerate(combined_roc_data.items()):
        mean_fpr, mean_tpr, mean_auc = average_roc(roc_data)
        plt.plot(mean_fpr, mean_tpr, color=colors(idx / len(combined_roc_data)), lw=4, 
                 label=f'{word} words (AUC = {mean_auc:.3f})') 

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=28, labelpad=20)  
    plt.ylabel('True Positive Rate', fontsize=28, labelpad=20) 
    plt.title('Average ROC Curves for Different Word Counts', fontsize=32, pad=20)  
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tick_params(axis='both', which='major', labelsize=24)

    plt.legend(loc='lower right', fontsize=24, frameon=False, framealpha=0.9, handletextpad=2.0) 

    output_filename = f'data/roc.pdf'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)