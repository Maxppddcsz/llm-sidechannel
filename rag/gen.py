import os
import json
import random

# 定义填充和截取函数
def pad_or_truncate(doc, target_length):
    tokens = doc.split()
    current_length = len(tokens)
    
    if (current_length < target_length):
        # 填充文档，随机选择文档中的词进行重复
        while len(tokens) < target_length:
            tokens.append(random.choice(tokens))
    elif (current_length > target_length):
        # 截取文档
        tokens = tokens[:target_length]
    
    return ' '.join(tokens)

# 读取文件并分割文档
file_path = 'val.src.cleaned'

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

documents = []
for line in lines:
    split_docs = line.strip().split('|||||')
    documents.extend(split_docs)

# 计算每个文档的长度
document_lengths = [len(doc.split()) for doc in documents]

# 定义目标长度
target_lengths = [550]
target_docs = {length: [] for length in target_lengths}

# 分配文档到对应的数组中
for doc, length in zip(documents, document_lengths):
    for target_length in target_lengths:
        if target_length - 25 <= length <= target_length + 25 and len(target_docs[target_length]) < 500:
            target_docs[target_length].append(doc)
            break

for l in target_lengths:
# 如果1600 words的文档不足100个，则合并文档生成新的1600 words文档
    if len(target_docs[l]) < 100:
        combined_docs = []
        candidate_docs = [doc for doc in documents if l-120 <= len(doc.split()) <= l-80]

        while len(candidate_docs) >= 2 and len(target_docs[l]) < 100:
            doc1 = candidate_docs.pop(0)
            doc2 = candidate_docs.pop(0)
            combined_doc = doc1 + " " + doc2
            combined_tokens = combined_doc.split()

            # 截取生成的文档以确保长度为1600 words
            if len(combined_tokens) > l:
                combined_tokens = combined_tokens[:l]

            target_docs[l].append(' '.join(combined_tokens))

# 对每个文档数组进行处理
for target_length in target_docs:
    processed_docs = []
    for doc in target_docs[target_length]:
        processed_doc = pad_or_truncate(doc, target_length)
        processed_docs.append(processed_doc)
    
    target_docs[target_length] = processed_docs

# 创建目录并将文档保存为单独的JSON文件
for length in target_docs:
    directory = f'length_{length}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i, doc in enumerate(target_docs[length]):
        output_file = os.path.join(directory, f'doc_{i+1}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"0": doc,"query":"Summary the news","chunk_num":1}, f, ensure_ascii=False, indent=4)
            # print(len(doc.split()))
    
    print(f"{length} words文档已保存到目录 {directory}")
