# bottleneck.
import pickle

# given the number of the attack request, try to select as much as you can.

# in commonplace, the orthogonal group won't exceed 10.


for i in range(1, 10):
    # select different number.
    result_data = [0, 0, 0, 0]
    for j in range(0, 100):
        try:
            with open(f'Your_own_path_to/result_{j}_{i}', 'rb') as file:
                new_data = pickle.load(file)
                result_data[0] += new_data[0]
                result_data[1] += new_data[1]
                result_data[2] += new_data[2]
                result_data[3] += new_data[3]
        except FileNotFoundError:
            for k in range(i, 0, -1):
                try:
                    with open(f'Your_own_path_to/result_{j}_{k}', 'rb') as file:
                        new_data = pickle.load(file)
                        result_data[0] += new_data[0]
                        result_data[1] += new_data[1]
                        result_data[2] += new_data[2]
                        result_data[3] += new_data[3]
                    break
                except FileNotFoundError:
                    continue
                
    if i == 1:
        print(f'total number of samples: {result_data[2]}')
    
    print(f'maximum select {i} sentences')
    print(f'fp case: {result_data[0]} tp case: {result_data[1]}  fp_test_time: {result_data[2]} tp_test_time: {result_data[3]} ')



