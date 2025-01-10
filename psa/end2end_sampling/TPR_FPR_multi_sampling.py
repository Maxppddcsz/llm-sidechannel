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

model_id = f"YOUR_OWN_PATH_TO_THE_MODEL/{model}"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select the prompts that the Attacker has.
# Get used_contents.
file_path = "YOUR_OWN_PATH_TO_THE_DATASET/YOUR_DATASET.json"
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

used_contents = text_contents[:1000]


# attacker prompt length
att_start_length = 50


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
# For Single Token Case.
# In SGLANG source code, we found that the last token of the prompt won't be sent to the radix cache for prefix-matching, so we can add a dummy token, we choose 'a' as token for example.
for i in tqdm(range(400), desc="Requests"):
    # Assume by statistically way, the template and several initial tokens of the system prompts are already guessed right.
    system_prompt = used_contents[i]
    triggering_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nMYGO [/INST]"
    
    encoding = tokenizer(triggering_prompt, truncation=False, padding=False, return_tensors="pt")

    # We first consider recovering only one token or two tokens here.
    # while att_start_length < 10:
    pos_ids = encoding["input_ids"][:, :att_start_length + 1]
    
    # Copy pos_ids to neg_ids and modify the last token
    neg_ids = pos_ids.clone()
    if neg_ids[0, -1] == 0:  # If the last token is a padding token, handle accordingly
        neg_ids[0, -1] = 1  # Replace with a token different from padding
    else:
        neg_ids[0, -1] = (neg_ids[0, -1] + 1) % tokenizer.vocab_size  # Replace last token with a different one
    
    neg_ids_v2 = pos_ids.clone()
    if neg_ids_v2[0, -1] == 0:  # If the last token is a padding token, handle accordingly
        neg_ids_v2[0, -1] = 1  # Replace with a token different from padding
    else:
        neg_ids_v2[0, -1] = (neg_ids_v2[0, -1] + 2) % tokenizer.vocab_size  # Replace last token with a different one

    # Add a dummy token at the end, which won't be taken into consideration when it comes to the token prefilling.
    pos_sentence = tokenizer.decode(pos_ids[0], skip_special_tokens=True) + " a"
    neg_sentence = tokenizer.decode(neg_ids[0], skip_special_tokens=True) + " a"
    neg_sentence_v2 = tokenizer.decode(neg_ids_v2[0], skip_special_tokens=True) + " a"

    check_result = []
    check_fpr_result = []
    
    # RECORD 10 VALUES, USING POS_SENTENCE TO
    # ADJUST CONCURRENT THRESHOLD.
    # DYNAMIC THRESHOLD CAN BE ENSURED USING THE 
    # AVERAGE VALUE, BUT HERE WE CAN ALSO SIMPLY COMPARE
    # THE POS/NEG CASES.
    # ====================================================
    for i in tqdm(range(10), desc="Prompts"):
        complete(triggering_prompt)

        time.sleep(0.3)
        latency1 = get_ttft(pos_sentence)
        #fp.write(f"{latency}\n")

        latency2 = get_ttft(pos_sentence)
        #fn.write(f"{latency}\n")

        latency3 = get_ttft(neg_sentence)

        if latency2 - latency1 > 0.0010:
            check_result.append(0)
        else:
            check_result.append(1)

        if latency3 - latency1 > 0.0010:
            check_fpr_result.append(0)
        else:
            check_fpr_result.append(1)
    
        flush_cache()
    
    # SUM UP THE RESULT.
    if sum(check_result) >= 5:
        TPR += 1

    if sum(check_fpr_result) >= 5:
        FPR += 1

# 4000 SAMPLES WILL BE SAMPLED, BUT THEM ONLY FORM 400 IN FACT.
# BECAUSE WE USE 10 SAMPLES FOR EACH PREDICATED REQUESTS.
print(f'TPR: {TPR/400}, FPR: {FPR/400}.\n')

fn.close()
fp.close()
fn2.close()
