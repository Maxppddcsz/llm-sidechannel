import csv
import os
import time
import re
import random
import argparse

from sentence_transformers import SentenceTransformer, util

from datasets import load_dataset
from gptcache.similarity_evaluation import OnnxModelEvaluation, SbertCrossencoderEvaluation

from names_dataset import NameDataset, NameWrapper

import networkx as nx

from tqdm import tqdm

import pickle

import networkx as nx

from gptcache.embedding import Huggingface

from sklearn import preprocessing

import numpy as np

import copy

onnx = OnnxModelEvaluation()


# False negative means the attack sequence has smaller score with all of the victim
def similarity_evaluation_onnx(attack, victim, thres):
    '''
    For the similarity evaluation
    '''
    score = onnx.evaluation(
        {
            'question': attack
        },
        {
            'question': victim
        }
    )
    # No matter what kind of questions the victim inputs
    # if the score is higher than the thres
    # we return true
    if score > thres:
        return True
    else:
        return False

def hit_calculate(False_label, True_label, attack_sentences, thres, result_id):

    tp_test_time = 0
    fp_test_time = 0

    True_samples = True_label
    False_samples = False_label

    new_true_samples = copy.deepcopy(True_samples)
    new_false_samples = copy.deepcopy(False_samples)

    tpr_result = 0
    fpr_result = 0
        
    # If more than half trial is TRUE, we consider the prediction is TRUE.
    # assume the attack_sentences already are calculated then.
    for i in range(len(attack_sentences)):
        # the True_label and False_samples number will decrease,
        # False_samples:
        # if judge it to be hit, we add fpr_result. and remove the sentence from the samples.
        for j in range(len(False_samples)):
            if similarity_evaluation_onnx(False_samples[j], attack_sentences[i], thres):
                fpr_result += 1
                new_false_samples.remove(False_samples[j])
            fp_test_time += 1
                
        # Similar case for the True samples.
        for j in range(len(True_samples)):
            if similarity_evaluation_onnx(True_samples[j], attack_sentences[i], thres):
                tpr_result += 1
                new_true_samples.remove(True_samples[j])
            tp_test_time += 1

        True_samples = copy.deepcopy(new_true_samples)
        False_samples = copy.deepcopy(new_false_samples)

        current_result = [fpr_result, tpr_result, fp_test_time, tp_test_time]
        fr = open(f'Your_onw_path_to/result_{result_id}_{i+1}', 'wb')
        pickle.dump(current_result, fr)

        fr.close()
        

def greedy_ways(attack, thres):
    sentence_group = [attack[0]]
    for j in range(1, len(attack)):
        new_sentence = attack[j]
        
        flag = 0
        
        for vic in sentence_group:
            score = onnx.evaluation(
                {
                    'question': new_sentence
                },
                {
                    'question': vic
                }
            )
            # HIT at least for half of the victim
            # Then we consider it is HIT
            if score > thres:
                flag = 1
        
        if flag == 1:
            continue
        else:
            sentence_group.append(new_sentence)


    return sentence_group


total_length = 0
true_case = 0
false_case = 0

thres = 0.8

# use result to store the 
for i in tqdm(range(0, 100), 'processing:'):
    filename = f'Your_onw_path_to/data_{i}'
    fr = open(filename, 'rb')

    try:
         False_label = pickle.load(fr)
    except EOFError:
        continue
    
    True_label = pickle.load(fr)
    topk_seed = pickle.load(fr)

    if False_label == []:
        continue

    attack_sentences = greedy_ways(topk_seed, thres)

    
        
    hit_calculate(False_label, True_label, attack_sentences, thres, i)

