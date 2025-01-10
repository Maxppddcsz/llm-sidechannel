# Despite that we can't figure out the real value of the HIT and MISS, the differences can be calculated.
# We will record three groups, each containing 50 times record TTFT.

# This lab is done in a stable environment (less voltage)
# To draw pictures in Figure 3.

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

model_id = f"YOUR_OWN_PATH_TO_THE_TEST_MODEL/{model}"

tokenizer = AutoTokenizer.from_pretrained(model_id)


# Select the prompts that the Attacker has.
# Get used_contents.
file_path = "YOUR_OWN_PATH_TO_THE_TRAIN_DATASET/YOUR_FILE_NAME.json"

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

used_contents = text_contents[:400]


# attacker prompt length
# Change it if you want to get another samples of token positions.
att_start_length = 13


# flush the cache of sglang.
# In Sampling time, we use flush_cache to accelerate the process.
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
original = f"data/distributions/{model}/sampling/org_{att_start_length}.txt"

os.makedirs(os.path.dirname(negative), exist_ok=True)

fn = open(negative, 'w')
fp = open(positive, 'w')
fo = open(original, 'w')
# For Single Token Case.
# In SGLANG source code, we found that the last token of the prompt won't be sent to the radix cache for prefix-matching, so we can add a dummy token, we choose 'a' as token for example.
for i in tqdm(range(len(used_contents)), desc="Requests"):
    # Assume by statistically way, the template and several initial tokens of the system prompts are already guessed right.
    system_prompt = used_contents[i]
    triggering_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nMYGO [/INST]"
    
    encoding = tokenizer(triggering_prompt, truncation=False, padding=False, return_tensors="pt")

    # We first consider recovering only one token or two tokens here.
    # while att_start_length < 10:
    org_ids = encoding["input_ids"][:, :att_start_length]
    pos_ids = encoding["input_ids"][:, :att_start_length + 1]
    #org_ids = encoding["input_ids"][:, :att_start_length + 1] 
    # Copy pos_ids to neg_ids and modify the last token
    neg_ids = pos_ids.clone()
    if neg_ids[0, -1] == 0:  # If the last token is a padding token, handle accordingly
        neg_ids[0, -1] = 1  # Replace with a token different from padding
    else:
        neg_ids[0, -1] = (neg_ids[0, -1] + 1) % tokenizer.vocab_size  # Replace last token with a different one
    
    #neg_ids_v2 = pos_ids.clone()
    #if neg_ids_v2[0, -1] == 0:  # If the last token is a padding token, handle accordingly
    #    neg_ids_v2[0, -1] = 1  # Replace with a token different from padding
    #else:
    #    neg_ids_v2[0, -1] = (neg_ids_v2[0, -1] + 2) % tokenizer.vocab_size  # Replace last token with a different one

    # Add a dummy token at the end, which won't be taken into consideration when it comes to the token prefilling.
    pos_sentence = tokenizer.decode(pos_ids[0], skip_special_tokens=True) + " a"
    neg_sentence = tokenizer.decode(neg_ids[0], skip_special_tokens=True) + " a"
    org_sentence = tokenizer.decode(org_ids[0], skip_special_tokens=True) + " a"
    #neg_sentence_v2 = tokenizer.decode(neg_ids_v2[0], skip_special_tokens=True) + " a"

    print(f'org_sentence: {org_sentence} {org_ids[0]}\n')
    print(f'pos_sentence: {pos_sentence} {pos_ids[0]}\n')
    print(f'neg_sentence: {neg_sentence} {neg_ids[0]}\n')
    for i in tqdm(range(10), desc="Prompts"):
        complete(triggering_prompt)
        
        time.sleep(0.3)
        latency = get_ttft(org_sentence)
        fo.write(f"{latency}\n")
        latency = get_ttft(pos_sentence)
        fp.write(f"{latency}\n")
        
        latency = get_ttft(neg_sentence)
        fn.write(f"{latency}\n")


        #latency = get_ttft(neg_sentence_v2)
        #fn2.write(f"{latency}\n")

        flush_cache()

fo.close()
fn.close()
fp.close()
#fn2.close()
