from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
import os

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.9,max_model_len=None
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

llm.set_tokenizer(tokenizer)

output_dir = "data/"

# Create an tokenizer and LLM.
cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
cache_fuse_metadata['collect'] = False
cache_fuse_metadata['check'] = False

s_start_full = [733, 4138, 28793]
s_start_len = len(s_start_full) + 1

s_start = []
s_start_1_len = len(s_start) + 1

s_end = [518, 29914, 25580, 29962]
s_end = [733, 28748, 16289, 28793]
s_end_len = len(s_end)

old_kvs = []

def process_file(file_path):
    f = open(f"{file_path}")
    ex = json.load(f)
    chunk_num = ex['chunk_num']
    doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
    q_prompt = ex['query']
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    # Create an tokenizer and LLM.
    # cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    # cache_fuse_metadata['collect'] = False
    # cache_fuse_metadata['check'] = False

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)

    return doc_chunk_ids,input_prompt

def upload_file(doc_chunk_ids):

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False

    num_layer = 32
    chunk_past_key_values = []
    
    # Concatenate old KVs
    for i in range(len(doc_chunk_ids)):
        # print(doc_chunk_ids[i])
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
        outputs = llm.generate(prompts, sampling_params)
        # print(outputs)
        
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            if i == 0:
                temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
                temp_v = past_key_values[1][:s_start_len].clone()
            else:
                temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
                temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()    

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                #pdb.set_trace()
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
        #print(temp_k.shape[0])
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
    

print("====Warm up====")
sampling_params = SamplingParams(temperature=0, max_tokens=10)
for i in range(10):
    output = llm.generate("Just warming up...", sampling_params)


length_directories = [550]

for length in length_directories:
    doc_chunk_ids,input_prompt = process_file(f"length_{length}/doc_1.json")

    upload_file(doc_chunk_ids)
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    # for p in input_prompt:
    output = llm.generate([input_prompt], sampling_params)
    # print(output)
    print(f"Cached generation: {output[0].outputs[0].text}")
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")

    hit_times_file = open(os.path.join(output_dir, f"data/doc-hit_length_{length}.txt"), "w")
    miss_times_file = open(os.path.join(output_dir, f"data/doc-miss_length_{length}.txt"), "w")

    for i in range(500):
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        # cache_fuse_metadata["check"] = True
        # cache_fuse_metadata['collect'] = False
        output = llm.generate([input_prompt], sampling_params)
        # print(output)
        # print(f"Cached generation: {output[0].outputs[0].text}")
        response_time = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        hit_times_file.write(f"{response_time}\n")
        # print(f"TTFT with cache: {response_time}")
        

    print("====Attack====")
    for i in range(2,501):
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        doc_chunk_ids,input_prompt = process_file(f"length_{length}/doc_{i}.json")
        # cache_fuse_metadata["check"] =  False
        # cache_fuse_metadata['collect'] = False
        output = llm.generate([input_prompt], sampling_params)
        # print(output)
        # print(f"Cached generation: {output[0].outputs[0].text}")
        response_time = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        miss_times_file.write(f"{response_time}\n")