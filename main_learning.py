import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    param_sets_list = [
        {
            'n_steps': [5],
            'n_states': [3],
            'n_arms': [3, 4, 5],
            'transition_type': ['structured'],
            'utility_functions': [(1, 0), (2, 4), (3, 16)],
            'thresholds': [0.3, 0.5, 0.7],
            'arm_choices': [1]
        },
        {
            'n_steps': [5],
            'n_states': [3],
            'n_arms': [3, 4, 5],
            'transition_type': ['clinical', 'structured'],
            'utility_functions': [(1, 0), (2, 4), (3, 16)],
            'thresholds': [0.3, 0.5, 0.7],
            'arm_choices': [1]
        },
    ]

    learning_episodes = 50
    n_averaging_episodes = 5
    n_iterations = 10

    save_data = True
    PATH = f'./learning-finite-{learning_episodes}-{n_averaging_episodes}-{n_iterations}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Iterate through each parameter set in the list
    for param_sets in param_sets_list:
        print("=" * 50)
        print(f"Processing parameter set: {param_sets}") #Added to show the set being processed
        param_list = product(
            param_sets['n_steps'],
            param_sets['n_states'],
            param_sets['n_arms'],
            param_sets['transition_type'],
            param_sets['utility_functions'],
            param_sets['thresholds'],
            param_sets['arm_choices'],
            [learning_episodes],  #  Add the fixed parameters to the product
            [n_averaging_episodes],
            [n_iterations],
            [save_data],
            [PATH]
        )
        for params in param_list:
            print('='*50)
            print(params)
            run_learning_combination(params)

if __name__ == '__main__':
    main()
