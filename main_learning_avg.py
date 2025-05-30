import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    n_steps = 1000
    n_iterations = 5
    
    param_sets_list = [
        {
            'n_steps': [n_steps],
            'n_states': [5],
            'n_arms': [3, 4, 5],
            'transition_type': ['structured'],
            'arm_choices': [1]
        },
        {
            'n_steps': [n_steps],
            'n_states': [0],
            'n_arms': [3, 4, 5],
            'transition_type': ['clinical', 'clinical-v2', 'clinical-v3', 'clinical-v4'],
            'arm_choices': [1]
        },
    ]

    save_data = True
    PATH = f'./learning-average-{n_steps}-{n_iterations}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    for param_sets in param_sets_list:
        print("=" * 50)
        print(f"Processing parameter set: {param_sets}") #Added to show the set being processed
        param_list = product(
            param_sets['n_steps'],
            param_sets['n_states'],
            param_sets['n_arms'],
            param_sets['transition_type'],
            param_sets['arm_choices'],
            [n_iterations],
            [save_data],
            [PATH]
        )
        for params in param_list:
            print('='*50)
            run_avg_learning_combination(params)

if __name__ == '__main__':
    main()
