import argparse
import random

from datasets import load_dataset
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation

parser = argparse.ArgumentParser(description='Example of command args')

# the number of irrelevant sentences we want to generate
parser.add_argument('-n', '--num', type=int, default=10)

args = parser.parse_args()

num = args.num

sbert = SbertCrossencoderEvaluation()

thres = 0.3

ds = load_dataset("THUDM/LongWriter-6k")

def similarity_evaluation_sbert(attack, victim, thres):
    '''
    For the similarity evaluation
    '''
    score = sbert.evaluation(
        {
            'question': attack
        },
        {
            'question': victim
        }
    )

    if score > thres:
        return True
    else:
        return False
    

def new_sbert_sentence(new, old_groups):
    '''
    Generate a new sentence that is dissimilar to the victim sentence.
    '''
    for old in old_groups:
        if similarity_evaluation_sbert(new, old, thres):
            return False

    return True

result = []

for i in range(num):
    random_number = random.randint(0, 5999)
    another_random_number = random.randint(0, 5999)
    new_sentence = ds["train"][random_number]["messages"][1]["content"][0 + another_random_number:500 + another_random_number]
    
    if i == 0:
        result.append(new_sentence)
        continue

    while not new_sbert_sentence(new_sentence, result):
        random_number = random.randint(0, 5999)
        another_random_number = random.randint(0, 5999)
        new_sentence = ds["train"][random_number]["messages"][1]["content"][0 + another_random_number:500 + another_random_number]

    result.append(new_sentence)


print(result)
