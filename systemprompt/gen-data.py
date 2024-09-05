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

id = int(sys.argv[1])

words = get_english_words_set(['web2'], lower=True)
filtered_words = [word for word in words if 2 <= len(word) <= 6]

def warmup(prefix):
    warmup = []
    warmup_sentence_list = []
    while len(warmup) < 500:
        warmup_word = random.choice(filtered_words)
        # not equal to the hit case.
        if warmup_word not in warmup and warmup_word != 'math':
            warmup.append(warmup_word)
    
    for i in range(len(warmup)):
        #flush_cache()
        warmup_question = prefix + ' ' + warmup[i]
        # print(warmup_question)
        # start_time = time.time()
        initial_states = few_shot_mmlu.run(question=warmup_question)
        initial_states["answer"].strip()
        # end_time = time.time()
        warmup_sentence_list.append(warmup_question)
        # print(f'warmup latency: {end_time - start_time}')
    return warmup_sentence_list

def first_n_words(sentence,n):
    words = re.findall(r'\b\w+\b|\S', sentence)
    return ' '.join(words[:n])

prompts = []
with open(f'/mnt/data1/pzx/sglang/lab/code/hit/new_prompt_{id}.txt', 'r') as file:
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

negative = f"data/prefix_{id}.txt"
positive = f"data/prefix_{id+1}.txt"

fn = open(negative, 'w')
fp = open(positive, 'w')

# MISS Case, from prompts
# 4k samples: select 40 different miss case, each running for 100 times.
miss_prompts = []
for _ in range(40):
    new_prompt = prompts[random.randint(0, len(prompts)-1)]
    miss_prompts.append(new_prompt)

for i in tqdm(range(10), desc="Round"):
    warmup_word = random.choice(filtered_words)
    # miss prompt => miss 2.
    mp = first_n_words(miss_prompts[i], id+2)
    # hit prompt => hit first, miss second.
    hp = first_n_words(original_prompt, id+1) + " " + warmup_word
    
    print(f'mp: {mp} hp: {hp} original: {original_prompt}')
    
    # prefix = first_n_words(original_prompt,id)
    # print(f'prefix: {prefix}')
    # warmup(prefix=prefix)

    # test mp for 100 times.
    for m in tqdm(range(1000), 'Processing:'):
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


    for n in tqdm(range(1000), 'Processing'):
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
