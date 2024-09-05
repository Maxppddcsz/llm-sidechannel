# Compose a meeting agenda for an interdisciplinary team discussing the treatment plan for [Name] with [medical condition]

import csv
import os
import time
import re
import random
import argparse

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from datasets import load_dataset
from gptcache.similarity_evaluation import OnnxModelEvaluation, SbertCrossencoderEvaluation

from names_dataset import NameDataset, NameWrapper

import networkx as nx

from tqdm import tqdm

import pickle


client = OpenAI(
    base_url="https://api.ai.cs.ac.cn/v1",
    api_key="sk-363kh3zATMx4rt1xIDCUvSUbTSCMrMNwZkYo9H8V6h9fwLE4",
)

# The library takes time to initialize because the database is massive. A tip is to include its initialization in your app's startup process.
nd = NameDataset()
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

def similar_eval(attack, victim):
    score = onnx.evaluation(
        {
            'question': attack
        },
        {
            'question': victim
        }
    )
    return score

# [Tested]
# Given the system_prompt and the question, return the string here.
def prompt(system_prompt, question):
    info = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        model="gpt-3.5-turbo",
    )

    return info.choices[0].message.content


# [Tested].
# Given the question from the dataset, summarize the [medical condition]
# Use the word in the sentence without understanding.
def medical_summary(question):
    system_prompt = "Assume you are an experienced doctor. After reading the user's prompt, \
        summarize the question using the name of the relevant disease mentioned in the user's inquiry to determine what specific illness the user is asking about. \
        Always return with Phrase or word extracted from the question sentence, make sure the phrase or the word is the illness itself, not the treatment or other things."
    
    return prompt(system_prompt, question)


def num_string_to_sentenceslist(info):
    pattern = r'^\d+[.]\s.*$'
    seeds = []
    temp = info.splitlines()
    
    for line in temp:
        if re.match(pattern, line):
            line = re.sub("^\d+[.]\s", "", line)
            seeds.append(line)
    return seeds

# [Tested]
# Given the Tempalte of the phenomenon
def template_cluster():
    template = "Compose a meeting agenda for an interdisciplinary team discussing the treatment plan for {name} with {medical_condition}"

    system_prompt = "Assume you are an experienced Medical Scribe. Now what you will get from the prompt is a template, the bracket part {} records the keyword that you can not change.\
        Your duty is to paraphrase the template into different forms, provide me 80 semantic similar sentences and return it back. \
        All of the template you provide should have {name} and {medical_condition} part that is unchangeable."


    info = prompt(system_prompt, template)

    return num_string_to_sentenceslist(info)


def matrix(sentences):
    matrix = [[] for _ in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            score = onnx.evaluation(
                {
                    'question': sentences[i]
                },
                {
                    'question': sentences[j]
                }
            )
            matrix[i].append(score)
    
    return matrix

def orthognal_group(attack, num, thres):
    sentence_group = []
    cursor = 0
    
    # if the last_history don't change for a while.
    # then, we simply return the function beforehand.
    last_history = []
    while cursor < num:
        # judge whether the history of cursor don't change or not for at least 4 loops.
        # If yes, simply exit early.
        history_flag = 0
        for i in range(len(last_history)):
            if last_history[i] != cursor:
                history_flag = 1
        if history_flag == 0 and len(last_history) >= 4:
            sentence_group = []
        
        new_sentence = attack[random.randint(0, len(attack) - 1)]

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
        
        if flag == 0:
            last_history = []
            sentence_group.append(new_sentence)
            cursor += 1
        else:
            last_history.append(cursor)

    return sentence_group, num



# [Core Function]:
# Select the target and extend the target_seeds. 
# Assume the name is equal, others not.
def target_select_and_extend(name_set, ques, answ, orig_template, template_seeds, thres):
    name = name_set[random.randint(0, len(name_set) - 1)]
    pair = random.randint(0, len(ques) - 1)
    medical = ques[pair]
    
    target = orig_template.format(name=name, medical_condition=medical)

    # Use target_seeds as Label 1.
    target_seeds = list(map(lambda x: x.format(name=name, medical_condition=medical), template_seeds))

    target_coins = list(filter(lambda x: similarity_evaluation_onnx(x, target, thres) == True, target_seeds))

    seed = target_coins[:len(target_coins) // 5]
    True_label = target_coins[len(target_coins) // 5:]

    False_label = []
    for i in range(len(ques)):
        if ques[i] != medical:
            False_label.append(orig_template.format(name=name, medical_condition=ques[i]))
    
    length = min(len(False_label), len(True_label))
    
    False_label = False_label[:length]
    True_label = True_label[:length]
    # True Label: Seeds
    # The only thing in common is the name.

    
    # TestResult.
    true_result = list(map(lambda att: list(map(lambda vic: similar_eval(att, vic), True_label)), seed))
    false_result = list(map(lambda att: list(map(lambda vic: similar_eval(att, vic), False_label)), seed))

    return true_result, false_result


# ds = load_dataset("lavita/MedQuAD")
# ques = ds["train"]["question"]

# for i in range(10):
#     question = ques[random.randint(0, len(ques))]
#     summary = medical_summary(question)
#     print(f"the question: {question}, the sumamry: {summary}\n")

ds = load_dataset("lavita/MedQuAD")
ques = ds["train"]["question"]
answ = ds["train"]["answer"]

names = nd.get_top_names(n=5, country_alpha2='US')
name_set = names['US']['M'] + names['US']['F']

orig_target = "Compose a meeting agenda for an interdisciplinary team discussing the treatment plan for {name} with {medical_condition}"
# randomly select one of the name, and one of the question


# count = []
# for i in range(10):
#     seeds = template_cluster()

#     print(f'seeds:{seeds}\n')
#     x,y = target_select_and_extend(name_set, ques, answ, orig_target, seeds)
#     print(x)
#     print(y)

# print(count)
# print(sum(count)/10)

# select 10 names and 10 pairs before hand.
names = nd.get_top_names(n=10, country_alpha2='US')
name_set = names['US']['M'] + names['US']['F']

thres_group = [0.6, 0.7, 0.8, 0.9]

for thres in thres_group:
    ft = open(f'true_data_{thres}_One', 'wb')
    ff = open(f'false_data_{thres}_One', 'wb')

    for i in range(100):
        random.shuffle(name_set)
        subname_set = name_set[:10]
        pairs = []
        while len(pairs) < 20:
            new_pair = random.randint(0, len(ques) - 1)
            if new_pair not in pairs:
                pairs.append(new_pair)

        ques = list(map(lambda x: medical_summary(ques[x]), pairs))
        if None in ques:
            continue
        answ = list(map(lambda x: answ[x], pairs))

        true_result, false_result = target_select_and_extend(name_set, ques, answ, orig_target, template_cluster(), thres)

        true_result = list(map(lambda x: x / len(true_result), list(map(sum, zip(*true_result)))))
        false_result = list(map(lambda x: x / len(false_result), list(map(sum, zip(*false_result)))))

        print(true_result)
        print(false_result)
        pickle.dump(true_result, ft)
        pickle.dump(false_result, ff)
        
        # print('This loop is done.')
    
    ft.close()
    ff.close()

