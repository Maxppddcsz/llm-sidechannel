import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import re
import numpy as np
from tqdm import tqdm, trange
import pickle
from collections import Counter
import sys

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
def temperature_scaling(token_probs, temper, adjust, penalty):
    adjust_tokens = Counter(adjust)
    # print(f'adjust_tokens info: {adjust_tokens}')
    for key, value in adjust_tokens.items():
        for _ in range(value):
            token_probs[key[0]] /= penalty
    
    # Convert to logits
    logits = np.log(token_probs)
    
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


def adjust_prob(tokens_prob, temper, hist_dict, pos, log, penalty_table):
    # according to pos, select log and penalty from hist_dict and penalty_table.
    
    # select log that is the same length to (pos + 1) from hist_dict
    
    # first filter with length. second filter with log.
    adjust = dict(map(lambda x: (x[0][pos:pos + 1], x[1]), dict(filter(lambda x: tuple(log) == x[0][:pos], dict(filter(lambda x: len(x[0]) == pos + 1, hist_dict.items())).items())).items()))
    
    # print(f'adjust: {adjust}')
    tokens_prob = temperature_scaling(tokens_prob, temper, adjust, penalty_table[pos])

    return tokens_prob

def pace(prefix, step, history, temper, penalty_table):
    # New log for this pace
    log = []
    log_token = []
    for i in range(step):
        # print(f'i: {i}')
        # generate next token.
        tokens_prob, tokens = next_token_gpt(prefix)
        
        # Adjust the probs based on the history.
        hist_dict = Counter(history)
        # print(f'hist_dict: {hist_dict}')
        tokens_prob = adjust_prob(tokens_prob, temper, hist_dict, i, log, penalty_table)

        # select the first token
        choice = selected(tokens_prob)

        # new_token for the first token
        new_token = tokens[choice]

        prefix = prefix + new_token

        log.append(choice)
        log_token.append(new_token)

    return log, log_token


def oracle_check(startlen, oracle, step, log):
    if oracle[startlen: startlen + step] == log:
        return True
    
    return False


# For different prefix token length, we choose different classifier.
# return the result with FP, FN, TP times in test_time.
def test_ok(prefix, startlen, temper, test_time, oracle, step, penalty_table, tpr_fpr):
    # result: FP FN TP test_time
    result = [0, 0, 0, 0]

    count = 0
    
    # We will transfer history to Counter 
    history = []

    # scaled_tokens_prob, all_tokens = next_token_gpt(prefix)
    tpr, fpr = tpr_fpr[step - 1]

    check_time = 0

    if step == 1:
        check_time = 100
    else:
        check_time = 20

    for _ in range(test_time):
        count += 1
        # print('============ Guess Time =================')
        log, log_token = pace(prefix, step, history, temper, penalty_table)
        history += ([tuple(log[:i+1]) for i in range(len(log))])

        # print(f'history: {history}')
        # print(f'log: {log} log_token: {log_token}')
        # print(f'oracle_parital: {oracle[startlen: startlen + step]}')

        judge_list = []
        
        # Test for 50 times. 
        for _ in range(check_time):
            # CHANGE THE ORACLE JUDGE WITH YOUR CLASSIFIER
            # =====================================================
            # How to simulate the ORACLE ?
            # =====================================================
            # E.G.
            # If TPR is 0.63, FPR is 0.35, check_time == 50
            # 1. if the predicted token is equal to the true token. We have 90% to Figure it out, treat it as the answer we want.
            # 2. if the predicted token is not equal to the true token, we have 20% to judge it to be the right answer.
            #
            # Calculate:
            # - predicated == true token, 50 times => more than 25 times HIT => 96% to make it right.
            # - predicated != true token, 50 times => more than 25 times MISS => 99% to make it right.
            #
            # Assume the ORACLE always do have the same precision.
            # In reality, the ORACLE for larger step will always more precise.
            if oracle_check(startlen, oracle, step, log_token):
                if np.random.rand() > (1-tpr):
                    judge_list.append(1)
                else:
                    judge_list.append(0)
            else:
                if np.random.rand() <= fpr:
                    judge_list.append(1)
                else:
                    judge_list.append(0)

        value = sum(judge_list)

        if value >= 0.5 * len(judge_list):
            # FP
            if oracle[startlen: startlen + step] != log_token:
                # result: FP FN TP test_time
                print('############## FP ############')
                result[0] += 1
                # result[3] = count
                return result
            # TP
            else:
                print('############## TP ############')
                result[2] += 1
                result[3] = count
                return result
            #return count, True
        else:
            if oracle[startlen: startlen + step] == log_token:
                # print(f'false negative: {new_data}')
                print('############## FN ############')
                result[1] += 1
        

    # result[3] = test_time

    # Failed to find the result in 'test_time' attempts.
    return result


# For different prefix token length, we choose different classifier.
def select_next(scaled_probs, temper):
    choice = selected(scaled_probs)
    
    scaled_probs = temperature_scaling(scaled_probs, temper, choice)
    
    return choice, scaled_probs


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
# change the temperature with great variety
# at least, temperature 0.5 & 1.0 has relatively good results.
tempers = [0.5]

# select value below 4.
step = int(sys.argv[1])

test_time = 80

# result should be stored here.
fw = open(f'result_step{step}', 'wb')


# warming up

# select the first ten sentences.
# random.shuffle(mylist)

# mylist = mylist[:10]

# penalt table for different guessing steps.
penalty_table = [1.1, 1.2, 1.3, 1.4]

# tpr/fpr from ROC curves for different diffreneces of shared tokens.
tpr_fpr = [[0.56, 0.42], [0.8, 0.2], [0.83, 0.18], [0.85, 0.15]]

# Test whole for the list, do it once at first.naoz
# 0.5 - 1.0 are relative good temperature
for d in tqdm(range(len(tempers)), desc='temper:'):
    # randomly select one of the sentences from the dataset.
    # for k in tqdm(range(10), desc='requests:'):
    for k in tqdm(range(len(mylist)), desc='requests:'):
        victim = mylist[k]
        victim_token = process_string(victim)

        # print(f'victim token: {victim_token}')
        start = victim_token[0] + victim_token[1] + victim_token[2]
        startlen = 3
        print(f'$$$$$$ {k} $$$$$$')

        for count in range(1):
            # test the next token.
            result = test_ok(start, startlen, tempers[d], test_time, victim_token, step, penalty_table, tpr_fpr)
            pickle.dump(result, fw)
            # print(f'result info: {result}')
            if result[2] == 1:
                for i in range(step):
                    start += victim_token[startlen + i]
                startlen += step
            else:
                # print(f'{k}: end {count} here.\n')
                break
    
    
