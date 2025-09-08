import requests
import json
import time


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

url = "http://localhost:5000/"  # 如果服务端运行在其他机器上，需替换成对应的IP地址

while True:
    text_to_send = input("Welcome! Please enter your question (enter 'quit' to exit): ")
    if text_to_send == "quit":
        break
    data = {'text': text_to_send}
    headers = {'Content-Type': 'application/json'}
    try:
        ttft_value, response_text = get_ttft_and_content(url, data, headers)
        print(f"text: {response_text}")
        print(f"\033[32mttft: {ttft_value} s\033[0m")
    except requests.exceptions.RequestException as e:
        print("请求出错:", e)

