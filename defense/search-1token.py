import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import random
import re

import numpy as np
import time
from tqdm import tqdm, trange
import sglang as sgl

import joblib

import requests

import pickle
# TODO:
# DOWNLOAD THE MODEL FIRST, PLEASE CHANGE IT TO THE MODEL PATH
# ===============================================================
# Assume the Oracle has already been trained.

model_name = "Eidos777/sysprompt_judge_v0.001"

model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# predict next token of the prompt for at most num cases.
def next_token_gpt(prompt):
    # Encode input text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Get model outputs
    outputs = model(input_ids)

    next_token_logits = outputs.logits[:, -1, :]

    # Convert logits to probabilities
    next_token_probs = F.softmax(next_token_logits, dim=-1)

    all_token_ids = torch.arange(next_token_probs.size(-1))
    
    all_token_probs = next_token_probs.squeeze().tolist()

    all_tokens = [tokenizer.decode([token_id]) for token_id in all_token_ids]
    
    return all_token_probs, all_tokens

# the history here is a dictionary that store the current prefix test history on the miss time.
def temperature_scaling(all_token_probs, temper, last_miss):
    # last time
    #print(f'what is :{last_miss}')
    #print("ok")
    if last_miss != -1:
        all_token_probs[last_miss] /= 1.5
    
    # Convert to logits
    logits = np.log(all_token_probs)
    
    # Apply temperature scaling
    scaled_logits = logits / temper
    
    # Convert back to probabilities
    scaled_probs = np.exp(scaled_logits)
    
    # Normalize to ensure the sum is 1
    scaled_probs /= np.sum(scaled_probs)

    return scaled_probs

def selected(scaled_probs):
    options = np.arange(len(scaled_probs))

    choice = np.random.choice(options, p=scaled_probs)

    return choice


# For different prefix token length, we choose different classifier.
# return the result with FP, FN, TP times in test_time.
def test_ok(all_tokens, all_token_probs, temper, test_time, oracle):
    # result: FP FN TP test_time
    result = [0, 0, 0, 0]

    # assume the step is one, therefore we don't have to separate memory to store it.
    last_time = -1

    scaled_probs = all_token_probs
    
    count = 0
    
    for _ in range(test_time):
        count += 1
        # select the new token
        last_time = select_next(scaled_probs, temper, last_time)
        judge_list = []
        

        for _ in range(100):
            if all_tokens[last_time] == oracle:
                if np.random.rand() > 0.45:
                    judge_list.append(1)
                else:
                    judge_list.append(0)
            else:
                if np.random.rand() <= 0.42:
                    judge_list.append(1)
                else:
                    judge_list.append(0)

        value = sum(judge_list)

        if value >= 0.5 * len(judge_list):
            # FP
            if all_tokens[last_time] != oracle:
                # result: FP FN TP test_time
                result[0] += 1
                # result[3] = count
                print('############## FP ############')
                return result
            # TP
            else:
                result[2] += 1
                result[3] = count
                print('############## TP ############')
                return result
            #return count, True
        else:
            if all_tokens[last_time] == oracle:
                # print(f'false negative: {new_data}')
                print('############## FN ############')
                result[1] += 1
        

    # result[3] = test_time
    # Failed to find the result in 'test_time' attempts.
    return result

# For different prefix token length, we choose different classifier.
def select_next(all_token_probs, temper, last_time):
    scaled_probs = all_token_probs
    scaled_probs = temperature_scaling(all_token_probs, temper, last_time)
    
    # CHANGE THE ORACLE JUDGE WITH YOUR CLASSIFIER
    # =====================================================
    choice = selected(scaled_probs)
    
    # choice will act as last_time if this is judged as false
    return choice


def process_string(input_str):
    pattern = r'\s*\w+|\s*[,.]'
    
    result_list = re.findall(pattern, input_str)
    
    return result_list




# Some basic value and settings.

# all of the parameters will be treated as hyper parameters.

# assume only if the random value selected is higher than the thres,
# can the oracle made the result right

ds = load_dataset("teilomillet/system_prompt")

mylist = ds['train']['prompt']

# thres: changed from 0.1 - 0.3
# first assume we have 500 chances to select.
# change the temperature from 2.0 - 20.0 - 200.0


thres = 0.1
# at least, temperature 0.5 & 1.0 has relatively good results.
tempers = [0.5]
test_time = 80
fw = open('result_step1', 'wb')

# warming up

# select the first ten sentences.
# random.shuffle(mylist)

# mylist = mylist[:10]

token_count = [3]
# Test whole for the list, do it once at first.
# 0.5 - 1.0 are relative good temperature
for d in tqdm(range(len(tempers)), desc='temper:'):
    # randomly select one of the sentences from the dataset.
    # for k in tqdm(range(10), desc='requests:'):
    for k in tqdm(range(len(mylist)), desc='requests:'):
        victim = mylist[k]
        victim_token = process_string(victim)
        print(f'$$$$$ {k} $$$$$$')
        # print(f'victim token: {victim_token}')
        start = victim_token[0] + victim_token[1] + victim_token[2]
        startlen = 3

        # FIRST STEP: INITIALIZE THE SGLANG
        # ==================================
        # USE THE VICTIM TO INITIALIZE THE SGLANG
        
        all_token_probs, all_tokens = next_token_gpt(start)
        # avoid the case that victim_token[startlen + 1] is not in the 

        # Do as much as you can
        for i in range(1):
            # First select the next_token
            all_token_probs, all_tokens = next_token_gpt(start)

            # print(f'oracle: {victim_token[startlen]}')
            # test the next token.
            result = test_ok(all_tokens, all_token_probs, tempers[d], test_time, victim_token[startlen])
            
            pickle.dump(result, fw)

            # if result[2] == 1:
            #     start += victim_token[startlen]
            #     startlen += 1
            # else:
                # print(f'{k}: end {count} here.\n')
                # break
    
    