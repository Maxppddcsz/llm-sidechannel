# run after all the multitoken case is done.
import pickle

def read_numbers(f):
    result = []
    while True:
        try:
            data = pickle.load(f)
            result.append(data)
        except EOFError:
            break

    return result

file1 = 'result_step1'
file2 = 'result_step2'
file3 = 'result_step3'
file4 = 'result_step4'

fw1 = open(file1, 'rb')
fw2 = open(file2, 'rb')
fw3 = open(file3, 'rb')
fw4 = open(file4, 'rb')

result1 = read_numbers(fw1)
result2 = read_numbers(fw2)
result3 = read_numbers(fw3)
result4 = read_numbers(fw4)

TP1 = len(list(filter(lambda x: x[2] == 1, result1)))
TP2 = len(list(filter(lambda x: x[2] == 1, result2)))
TP3 = len(list(filter(lambda x: x[2] == 1, result3)))
TP4 = len(list(filter(lambda x: x[2] == 1, result4)))

FP1 = len(list(filter(lambda x: x[0] == 1, result1)))
FP2 = len(list(filter(lambda x: x[0] == 1, result2)))
FP3 = len(list(filter(lambda x: x[0] == 1, result3)))
FP4 = len(list(filter(lambda x: x[0] == 1, result4)))

FN1 = sum(list(map(lambda x: x[1], result1)))
FN2 = sum(list(map(lambda x: x[1], result2)))
FN3 = sum(list(map(lambda x: x[1], result3)))
FN4 = sum(list(map(lambda x: x[1], result4)))

TESTTIME1 = sum(list(map(lambda x: x[3], result1)))
TESTTIME2 = sum(list(map(lambda x: x[3], result2)))
TESTTIME3 = sum(list(map(lambda x: x[3], result3)))
TESTTIME4 = sum(list(map(lambda x: x[3], result4)))

print(f'result1: {TP1}, {FP1}, {FN1}, {TESTTIME1}')
print(f'result2: {TP2}, {FP2}, {FN2}, {TESTTIME2}')
print(f'result3: {TP3}, {FP3}, {FN3}, {TESTTIME3}')
print(f'result4: {TP4}, {FP4}, {FN4}, {TESTTIME4}')