import requests
import json
import time
import random

from tqdm import tqdm
from gptcache.embedding import Onnx

# As soon as the first token is not blank.
def get_ttft_and_content(url, data, headers):
    start_time = time.perf_counter()
    response = requests.post(url, data=json.dumps(data), headers=headers)

    for line in response.iter_lines():
        if line:  
            end_time = time.perf_counter()
            break

    ttft = end_time - start_time
    return ttft, response.text


def eviction(prompts, url, headers, num):
    for i in range(num):
        new_question = prompts[i]
        data = {'text': new_question}
        ttft, response = get_ttft_and_content(url, data, headers)
        #print("Time consuming: {:.2f}s".format(ttft))
        #print(f'Answer: {response_text(response)}\n')


url = "http://localhost:5000/"  # 如果服务端运行在其他机器上，需替换成对应的IP地址

num = 5

thres = 0.8

prompts = ['Is Gilbert syndrome inherited ?', 'How many people are affected by alkaptonuria ?', 'Is Wolf-Hirschhorn syndrome inherited ?', 'What is (are) Glomerular Diseases ?', 'What is the outlook for Yellow fever ?']

while True:
    text_to_send = input("Welcome! Please enter your question (enter 'quit' to exit): ")
    
    data = {'text': text_to_send}
    headers = {'Content-Type': 'application/json'}
    if text_to_send == "quit":
        break
    elif text_to_send == "evict":
        print('\033[33m[EVICTION!!]\033[0m')
        eviction(prompts, url, headers, num)
        continue

    try:
        ttft_value, response_text = get_ttft_and_content(url, data, headers)
        print(f"text: {response_text}")
        print(f"\033[32mttft: {ttft_value} s\033[0m")
        if ttft_value < 1:
            print('\033[31m[HIT!!]\033[0m')
    except requests.exceptions.RequestException as e:
        print("请求出错:", e)

