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

# select based on starting position of difference.
id = int(sys.argv[1]) 

words = get_english_words_set(['web2'], lower=True)
filtered_words = [word for word in words if 2 <= len(word) <= 6]

# read prompts from the file.
prompts = []
with open(f'your_own_path_to/new_prompt_{id}.txt', 'r') as file:
    prompts = [line.strip() for line in file.readlines()]

prompts = prompts[:1000]
print(len(prompts))


def first_n_words(sentence,n):
    words = re.findall(r'\b\w+\b|\S', sentence)
    return ' '.join(words[:n])


@sgl.function
def few_shot_mmlu(s,question):
    s += question + sgl.gen("answer", temperature=0, max_tokens=1)

# flush the cache of sglang.
def flush_cache():
    Response = requests.get("http://localhost:10005/flush_cache")


def process_string(input_str):
    pattern = r'\s*\w+|\s*[,.]' 
    result_list = re.findall(pattern, input_str)
    
    return result_list

# We can use another system prompt apart from the dataset we choose => more robustness
# Or We can choose from the system_prompt datasets
original_prompt = (
    "You are a math tutor who helps students of all levels understand and solve mathematical problems. "
    "Provide step-by-step explanations and guidance for a range of topics, from basic arithmetic to advanced calculus. "
    "Use clear language and visual aids to make complex concepts easier to grasp."
)


runtime_1 = sgl.RuntimeEndpoint("http://localhost:10005")

sgl.set_default_backend(runtime_1)


negative = f"your_own_path_to/prefix_{id}.txt"
positive = f"your_own_path_to/prefix_{id+1}.txt"

fn = open(negative, 'w')
fp = open(positive, 'w')

# MISS Case, from prompts
# 4k samples: select 40 different miss case, each running for 100 times.
miss_prompts = []
for _ in range(40):
    new_prompt = prompts[random.randint(0, len(prompts)-1)]
    miss_prompts.append(new_prompt)

for i in tqdm(range(10), desc="Round"):
    
    victim_tokens = process_string(original_prompt)
    warmup_token = " " + random.choice(filtered_words)
    
    while warmup_token in victim_tokens:
        warmup_token = " " + random.choice(filtered_words)
    
    # miss prompt => miss 2.
    mp = first_n_words(miss_prompts[i], id+2)
    # hit prompt => hit first, miss second.
    hp = first_n_words(original_prompt, id+1) + warmup_token
    
    # Trick: add a random token will stably increase the differenece between Pos/Neg latency samples.
    # => In fact the shared token difference is still one, but the prompts are longer after appending new tokens at the end.
    print(f'mp: {mp} hp: {hp} original: {original_prompt}')
    

    for m in tqdm(range(2000), 'Processing:'):
        # input the sentence here first
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


    for n in tqdm(range(2000), 'Processing'):
        # input the sentence here first
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
