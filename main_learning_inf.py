import os
from utils import *
import warnings
warnings.filterwarnings("ignore")


def main():
    param_sets = {
        'discount_factors': [0.9],
        'n_steps': [100],
        'n_states': [3],
        'n_augmnt': [10],
        'n_arms': [3],
        'transition_type': ['structured'], # clinical
        'utility_functions': [(3, 16)],
        'thresholds': [0.5],
        'arm_choices': [1]
    }

    learning_episodes = 200
    n_averaging_episodes = 5
    n_iterations = 10

    save_data = True
    PATH = f'./learning-infinite-{learning_episodes}-{n_averaging_episodes}-{n_iterations}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    param_list = [
        (df, nt, ns, ng, na, tt, ut, th, nc, learning_episodes, n_averaging_episodes, n_iterations, save_data, PATH)
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
