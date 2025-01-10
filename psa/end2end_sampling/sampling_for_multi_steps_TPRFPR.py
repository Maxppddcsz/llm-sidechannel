# For 5.1 section, get TPR/FPR when the minimum sharing granulity is 1, 2, 3, 4 tokens.
# Using TPR/FPR here to do simulation in 5.1 section.
import sglang as sgl
from sglang import OpenAI
import time
import sys
from tqdm import tqdm
import string
import re
import random
import requests
import torch

import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import json

# model for tokenizer.
model = sys.argv[1]
step = int(sys.argv[2])
model_id = f"YOUR_OWN_PATH_TO_MODEL/{model}"

tokenizer = AutoTokenizer.from_pretrained(model_id)

file_path = "YOUR_OWN_PATH_TO/YOUR_DATASET.json"
# The preprocessing program here we use will analysis the json file in the following form.
# [
#      {"text": "system prompt1"},
#      {"text": "system prompt2"},
# ]


with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

text_contents = []
for entry in data:
    text = entry.get("text", "")
    text_contents.append(text)


random.shuffle(text_contents)

used_contents = text_contents[:100]


# attacker prompt length
att_start_length = 23


# Make sure the port is figured right.
# flush the cache of sglang.
def flush_cache():
    Response = requests.get("http://localhost:54321/flush_cache")



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


seperator = "[INST] <<SYS>>\n"

flush_cache()

negative = f"data/distributions/{model}/sampling/neg_{att_start_length}.txt"
positive = f"data/distributions/{model}/sampling/pos_{att_start_length}.txt"
negative_2 = f"data/distributions/{model}/sampling/neg2_{att_start_length}.txt"

TPR = 0
FPR = 0


os.makedirs(os.path.dirname(negative), exist_ok=True)

fn = open(negative, 'w')
fp = open(positive, 'w')
fn2 = open(negative_2, 'w')

average_diff = []
# For Single Token Case.
# In SGLANG source code, we found that the last token of the prompt won't be sent to the radix cache for prefix-matching, so we can add a dummy token, we choose 'a' as token for example.
for i in tqdm(range(len(used_contents)), desc="Requests"):
    # Assume by statistically way, the template and several initial tokens of the system prompts are already guessed right.
    system_prompt = used_contents[i]
    triggering_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nMYGO [/INST]"
    
    encoding = tokenizer(triggering_prompt, truncation=False, padding=False, return_tensors="pt")

    # We first consider recovering only one token or two tokens here.
    # while att_start_length < 10:
    pos_ids = encoding["input_ids"][:, :att_start_length + 1]
    
    # neg request.
    neg_ids = pos_ids.clone()  # Clone pos_ids to ensure the same length
    neg_ids_v2 = pos_ids.clone()

    for index in range(1, step + 1): 
        current_token_index = neg_ids.shape[1] - index  
        if neg_ids[0, current_token_index] == 0:  
            neg_ids[0, current_token_index] = 1  
        else:
            neg_ids[0, current_token_index] = (neg_ids[0, current_token_index] + 1) % tokenizer.vocab_size  

    
    for index in range(1, step + 1):  
        current_token_index = neg_ids_v2.shape[1] - index  
        if neg_ids_v2[0, current_token_index] == 0:  
            neg_ids_v2[0, current_token_index] = 1  
        else:
            neg_ids_v2[0, current_token_index] = (neg_ids_v2[0, current_token_index] + 2) % tokenizer.vocab_size  


    # Add a dummy token at the end, which won't be taken into consideration when it comes to the token prefilling.
    pos_sentence = tokenizer.decode(pos_ids[0], skip_special_tokens=True) + " a"
    neg_sentence = tokenizer.decode(neg_ids[0], skip_special_tokens=True) + " a"
    neg_sentence_v2 = tokenizer.decode(neg_ids_v2[0], skip_special_tokens=True) + " a"
    print(f'pos_sentence: {pos_sentence}\n')
    print(f'neg_sentence: {neg_sentence}\n')
    print(f'neg_sentence_v2: {neg_sentence_v2}\n')
    check_result = []
    check_fpr_result = []
    for i in tqdm(range(10), desc="Prompts"):
        complete(triggering_prompt)
        time.sleep(0.3)
        latency1 = get_ttft(pos_sentence)
        #fp.write(f"{latency}\n")

        latency2 = get_ttft(pos_sentence)
        #fn.write(f"{latency}\n")

        latency3 = get_ttft(neg_sentence)

        if latency2 - latency1 > 0.0010:
            TPR += 0
        else:
            TPR += 1

        if latency3 - latency1 > 0.0010:
            FPR += 0
        else:
            FPR += 1
        
    
        average_diff.append(latency2 - latency1)
        average_diff.append(latency3 - latency1)
        flush_cache()
    

# GET TPR/FPR.
print(f'TPR: {TPR/1000}, FPR: {FPR/1000}.\n')
print(f'average_diff: {sum(average_diff)/2000}\n')


fn.close()
fp.close()
fn2.close()
