import os
from utils import *
import warnings
warnings.filterwarnings("ignore")


def main():
    param_sets = {
        'n_steps': [5],
        'n_states': [5],
        'n_arms': [5],
        'transition_type': ['structured'], # clinical
        'utility_functions': [(1, 0)],
        'thresholds': [0.5],
        'arm_choices': [1]
    }

    learning_episodes = 500
    n_averaging_episodes = 10
    n_iterations = 100

    save_data = True
    PATH = f'./learning-finite-{learning_episodes}-{n_averaging_episodes}-{n_iterations}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    param_list = [
        (nt, ns, na, tt, ut, th, nc, learning_episodes, n_averaging_episodes, n_iterations, save_data, PATH)
        for nt in param_sets['n_steps']
        for ns in param_sets['n_states']
        for na in param_sets['n_arms']
        for tt in param_sets['transition_type']
        for ut in param_sets['utility_functions']
        for th in param_sets['thresholds']
        for nc in param_sets['arm_choices']
    ]
    
    for params in param_list:
        print('='*50)
        run_learning_combination(params)

if __name__ == '__main__':
    main()
