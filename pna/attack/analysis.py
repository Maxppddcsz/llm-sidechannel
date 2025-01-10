import pickle
import sys

from gptcache.similarity_evaluation import OnnxModelEvaluation, SbertCrossencoderEvaluation

onnx = OnnxModelEvaluation()
thres = 0.8

def similarity_evaluation_onnx(attack, victim, thres):
    '''
    For the similarity evaluation
    '''
    score = onnx.evaluation(
        {
            'question': attack
        },
        {
            'question': victim
        }
    )
    # No matter what kind of questions the victim inputs
    # if the score is higher than the thres
    # we return true
    if score > thres:
        return True
    else:
        return False
    

result = [0 for _ in range(20)]
result_fpr = [0 for _ in range(20)]
count = 0
orig_target = "Compose a meeting agenda for an interdisciplinary team discussing the treatment plan for {name} with {medical_condition}"

# Dataset.
with open(f'{sys.argv[1]}', 'rb') as file:
    while True:
        try:
            data_list = pickle.load(file)
            if isinstance(data_list, list):
                # The last sentence is from the victim
                victim = data_list[-2]
                victim_fpr = data_list[-1]
                if len(data_list) > 0:
                    count += 1
                # Take it as a whole for considering
                tpr_result = 0
                tp_test_time = 0
                for i in range(len(data_list[:-2])):         
                    if similarity_evaluation_onnx(victim, data_list[i], thres):
                        result[i+1] += 1 
                        break
                
                for i in range(len(data_list[:-2])):
                    # Similar case for the True samples.
                    if similarity_evaluation_onnx(victim_fpr, data_list[i], thres):
                        result_fpr[i+1] += 1
                        break

        except EOFError:
            break

file.close()
print(result)
print(result_fpr)
print(count)
