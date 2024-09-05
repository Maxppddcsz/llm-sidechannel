'''
python gen-roc-data.py 3
'''

import sglang as sgl
from sglang import OpenAI
import time
import sys
from tqdm import tqdm
import string
import re
from english_words import get_english_words_set
import random
import requests
import torch

known = int(sys.argv[1]) # token numbers of prefix

def first_n_words(sentence,n):
    words = re.findall(r'\b\w+\b|\S', sentence)
    return ' '.join(words[:n])

prompts = []
with open(f'data/new_prompt_{known}.txt', 'r') as file:
    prompts = [line.strip() for line in file.readlines()]


prompts = prompts[:2000]
print(len(prompts))

original_prompt = (
    "You are a math tutor who helps students of all levels understand and solve mathematical problems. "
    "Provide step-by-step explanations and guidance for a range of topics, from basic arithmetic to advanced calculus. "
    "Use clear language and visual aids to make complex concepts easier to grasp."
)

@sgl.function
def few_shot_mmlu(s,question):
    s += question + sgl.gen("answer", temperature=0, max_tokens=1)

# flush the cache of sglang.
def flush_cache():
    Response = requests.get("http://localhost:10005/flush_cache")


runtime_1 = sgl.RuntimeEndpoint("http://localhost:10005")

sgl.set_default_backend(runtime_1)

prefix_list = [1,2,4,8]

for prefix in prefix_list:
    negative = f"data/prefix_{known}-{prefix}token.txt"
    positive = f"data/prefix_{known+prefix}-{prefix}token.txt"

    fn = open(negative, 'w')
    fp = open(positive, 'w')

    miss_prompts = prompts

    for miss_prompt in tqdm(miss_prompts, desc="Processing"):
        mp = first_n_words(miss_prompt, known+prefix)        
        print(f'mp: {mp} \n original: {original_prompt}')
        
        torch.cuda.synchronize()
        states = few_shot_mmlu.run(question=original_prompt)
        states["answer"].strip()
        torch.cuda.synchronize()
            
        tic = time.perf_counter()
        states = few_shot_mmlu.run(question=mp)
        states["answer"].strip()
        torch.cuda.synchronize()
        tictok = time.perf_counter()

        latency = tictok - tic
            
        fn.write(f"{latency}\n")

        flush_cache()
        time.sleep(0.1)

    for _ in tqdm(range(2000),desc="Processing"):
        hp = first_n_words(original_prompt, known+prefix)
        print(f'hp: {hp} \n original: {original_prompt}')
        torch.cuda.synchronize()
        states = few_shot_mmlu.run(question=original_prompt)
        states["answer"].strip()
        torch.cuda.synchronize()

        tic = time.perf_counter()
        states = few_shot_mmlu.run(question=hp)
        states["answer"].strip()
        torch.cuda.synchronize()
        tictok = time.perf_counter()
            
        latency = tictok - tic
            
        fp.write(f"{latency}\n")

        flush_cache()
        time.sleep(0.1)

    time.sleep(10)
