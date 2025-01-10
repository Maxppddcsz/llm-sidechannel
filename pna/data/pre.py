import openai
import random
import pickle
from openai import OpenAI
from names_dataset import NameDataset
from datasets import load_dataset

# 设置 OpenAI API 密钥
openai_api_key="sk-363kh3zATMx4rt1xIDCUvSUbTSCMrMNwZkYo9H8V6h9fwLE4" # 这里是API密钥
openai_api_base="https://api.ai.cs.ac.cn/v1"

client = OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key,
)

# 从 NameDataset 中获取名字
nd = NameDataset()
names = nd.get_top_names(n=5, country_alpha2='US')
name_set = names['US']['M'] + names['US']['F']  # 从美国数据集中获取前 10 个男名和女名的组合

# 从 "lavita/MedQuAD" 数据集中加载问题
def load_medical_questions():
    dataset = load_dataset("lavita/MedQuAD")
    questions = dataset["train"]["question"]  # 从训练集中提取问题
    return questions

# Function 1: 使用 prompt 函数生成类似问题
def prompt(system_prompt, question):
    info = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        model="gpt-3.5-turbo",
    )

    return info.choices[0].message.content

# Function 2: 根据医疗问题获取疾病摘要
def medical_summary(question):
    system_prompt = (
        "Assume you are an experienced doctor. After reading the user's prompt, "
        "summarize the question using the name of the relevant disease mentioned in the user's inquiry. "
        "Always return the name of the illness itself, not the treatment or other elements."
    )
    return prompt(system_prompt, question)

# Function 3: 生成模板的不同变体
def template_cluster():
    system_prompt = (
        "Assume you are an experienced Medical Scribe. You will receive a template, and the bracket part {} represents keywords you cannot change. "
        "Your task is to paraphrase the template into 100 semantically similar sentences. "
        "All of the generated templates should retain {name} and {medical_condition} as placeholders."
    )
    template = "Compose a meeting agenda for an interdisciplinary team discussing the treatment plan for {name} with {medical_condition}"
    response = prompt(system_prompt, template)
    
    # 使用正则表达式或其他逻辑解析返回结果为句子列表
    return response.split("\n")

# 替换模板中的名字和疾病
def replace_name_and_condition(name, condition):
    template = "Compose a meeting agenda for an interdisciplinary team discussing the treatment plan for {name} with {medical_condition}"
    return template.format(name=name, medical_condition=condition)

# 将生成的问题保存到文件中
def save_to_file(filename, content):
    with open(filename, 'w') as f:
        for line in content:
            f.write(line + "\n")

# Main function to generate questions using API and save them to a file
def main():
    # 加载 MedQuAD 数据集中的医疗问题
    ques = load_medical_questions()

    # 随机选择一些名字和问题，生成模板变体
    results = []
    for i in range(5):
        random_name = random.choice(name_set)
        random_condition = random.choice(ques)
        
        # 调用 medical_summary 获取问题摘要
        summary = medical_summary(random_condition)

        # 使用模板生成语句
        final_template = replace_name_and_condition(random_name, summary)
        results.append(final_template)
    
    # 将结果保存到文件
    save_to_file("output.txt", results)

# 执行主函数
if __name__ == "__main__":
    main()
