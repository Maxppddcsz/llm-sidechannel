# Try to have a closer look at the log of the server.

# Test the whether flood acts just like flush does.
import sglang as sgl
from sglang import OpenAI
from transformers import AutoTokenizer
import time
import sys
from tqdm import tqdm
import string
import re
import random
import requests
import torch
import argparse
import os 

from datasets import load_dataset

# LongWriter-6k from TSINGHUA-DM.
# THUDM/LongWriter-6k in Huggingface
ds = load_dataset("THUDM/LongWriter-6k")


parser = argparse.ArgumentParser(description='Example of command args')


parser.add_argument('-mt', '--max_new_tokens', type=int, default=1)
parser.add_argument('-n', '--n_parallel', type=int, default=4)
parser.add_argument("-b", "--batch_size", type=int, default=14)
parser.add_argument("-m", "--model", type=str, default="Llama-2-70B-GPTQ")

args = parser.parse_args()


model_id = f"/home/songlk/{args.model}"

tokenizer = AutoTokenizer.from_pretrained(model_id)

@sgl.function
def few_shot_mmlu(s,question):
    s += question + sgl.gen("answer", temperature=0.9,max_tokens=args.max_new_tokens)

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:54321"))

def new_batch_list():
    batch_list = []
    for i in range(args.batch_size):
        new_dict = {}
        random_number = random.randint(0, 5999)
        new_dict["question"] = ds["train"][random_number]["messages"][1]["content"][:500]
        batch_list.append(new_dict)

    return batch_list


origin_prompt = ("You are a travel itinerary assistant. You will help users create personalized trip plans based on their preferences and input regarding destination, budget, interests, and time constraints. Ensure that each itinerary includes essential details, such as accommodation options, transportation methods, key attractions, dining options, and free-time activities. Consider factors like user preferences for pace, specific requests for cultural experiences, or outdoor adventures if mentioned. Use up-to-date information about the destinations and include safety tips where necessary. Make sure that each itinerary is well-balanced, reasonable in terms of time, and enjoyable for the user. Guidelines: 1. Always prioritize user-driven preferences for destinations and activities. 2. Deliver a balance between exploration and relaxation within the itinerary. 3. Offer insights into local culture and practices relevant to the destination. 4. Help users maximize value for money in booking and planning. 5. Create itineraries that bring joy and valuable experiences to users, taking into account family or individual travelers. You are a travel itinerary assistant. You will help users create personalized trip plans based on their preferences and input regarding destination, budget, interests, and time constraints. Ensure that each itinerary includes essential details, such as accommodation options, transportation methods, key attractions, dining options, and free-time activities. Consider factors like user preferences for pace, specific requests for cultural experiences, or outdoor adventures if mentioned. Use up-to-date information about the destinations and include safety tips where necessary. Make sure that each itinerary is well-balanced, reasonable in terms of time, and enjoyable for the user. Guidelines: 1. Always prioritize user-driven preferences for destinations and activities. 2. Deliver a balance between exploration and relaxation within the itinerary. 3. Offer insights into local culture and practices relevant to the destination. 4. Help users maximize value for money in booking and planning. 5. Create itineraries that bring joy and valuable experiences to users, taking into account family or individual travelers.")
length_list = [10, 100, 200, 400]


def cut_prompt(prompt, length_list):
    prompt_list = []
    encoding = tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
    for len in length_list:
        input_ids = encoding["input_ids"][:, :len]
        org_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        prompt_list.append(org_sentence)

    print(f'list: {prompt_list}')
    return prompt_list, prompt_list[0]

def test_request(text):
    response = requests.post(
        "http://localhost:54321/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
            },
            # "stream": True,
        },
        #stream=True,
    )

# flush the cache of sglang.
def flush_cache():
    Response = requests.get("http://localhost:54321/flush_cache")

 
    # send_request(test_prompt) 

# testfile = f"/mnt/data1/pzx/sglang-lab/Eidos/evict/single_very_long_{args.max_new_tokens}_{args.n_parallel}.txt"
# fd = open(testfile, "a")
failed_times = 0
batch_list = new_batch_list()

prompt_list, pos_prompt = cut_prompt(origin_prompt, length_list)

flush_cache()

# Test the usability of eviction.
for i in tqdm(range(len(prompt_list)), desc="Round"):
    org_prompt = prompt_list[i]
    pos = f'data/evict/{args.model}_test/evict/time_{length_list[i]}.txt'
    os.makedirs(os.path.dirname(pos), exist_ok=True)
    
    fp = open(pos, "w")
    
    for m in tqdm(range(100), desc='Processing:'):
        states = few_shot_mmlu.run_batch(
            batch_list,
            temperature=0,
            progress_bar=True,
        )

        preds = []
        for i in range(len(states)):
            preds.append(states[i]["answer"])

        test_request(org_prompt)
        torch.cuda.synchronize()
        time.sleep(0.1)

        tic = time.time()
        test_request(pos_prompt)
        torch.cuda.synchronize()
        tok = time.time()

        fp.write(f'{tok - tic}\n')
    
    fp.close()
        
        


