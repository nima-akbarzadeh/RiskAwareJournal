import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    param_sets = {
        'discount_factors': [0.75, 0.9, 0.99],
        'n_steps': [5, 20],
        'n_states': [3, 5],
        'n_augmnt': [10],
        'n_arms': [5, 10],
        'transition_type': ['clinical', 'structured'], # clinical, structured
        'utility_functions': [(1, 0), (2, 4), (3, 16)],
        'thresholds': [0.2, 0.5, 0.7],
        'arm_choices': [1, 2]
    }

    learning_episodes = 500
    n_averaging_episodes = 10
    n_iterations = 100

    save_data = True
    PATH = f'./learning-nsfinite-{learning_episodes}-{n_averaging_episodes}-{n_iterations}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    param_list = [
        (df, nt, ns, ng, na, tt, ut, th, nc, learning_episodes, n_averaging_episodes, n_iterations, save_data, PATH)
        for df in param_sets['discount_factors']
        for nt in param_sets['n_steps']
        for ns in param_sets['n_states']
        for ng in param_sets['n_states']
        for na in param_sets['n_arms']
        for tt in param_sets['transition_type']
        for ut in param_sets['utility_functions']
        for th in param_sets['thresholds']
        for nc in param_sets['arm_choices']
    ]
    
    for params in param_list:
        print('='*50)
        run_ns_learning_combination(params)

if __name__ == '__main__':
    main()
