import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    n_steps = 1000
    discount = 0.99999
    param_sets = {
        'discount_factors': [discount],
        'n_steps': [n_steps],
        'n_states': [3], # 3 for clinical.
        'n_augmnt': [10],
        'n_arms': [2, 3],
        'transition_type': ['structured', 'clinical'], # structured, clinical
        'utility_functions': [(3, 4), (3, 16)],
        'thresholds': [0.1, 0.3, 0.5],
        'arm_choices': [1]
    }
    n_iterations = 100

    save_data = True
    PATH = f'./learning-infinite-{discount}-{n_steps}-{n_iterations}-test2/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    param_list = [
        (df, nt, ns, ng, na, tt, ut, th, nc, n_iterations, save_data, PATH)
        for df in param_sets['discount_factors']
        for nt in param_sets['n_steps']
        for ns in param_sets['n_states']
        for ng in param_sets['n_augmnt']
        for na in param_sets['n_arms']
        for tt in param_sets['transition_type']
        for ut in param_sets['utility_functions']
        for th in param_sets['thresholds']
        for nc in param_sets['arm_choices']
    ]
    
    for params in param_list:
        print('='*50)
        run_inf_learning_combination(params)

if __name__ == '__main__':
    main()
