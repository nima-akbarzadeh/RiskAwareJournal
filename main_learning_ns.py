import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    param_sets_list = [
        {
            'discount_factors': [0.9, 0.99],
            'n_steps': [5, 10],
            'n_states': [4, 5],
            'n_arms': [3, 4, 5],
            'transition_type': ['structured'],
            'utility_functions': [(1, 0), (2, 4), (3, 16)],
            'thresholds': [0.2, 0.5, 0.7],
            'arm_choices': [1]
        },
        {
            'discount_factors': [0.9, 0.99],
            'n_steps': [5, 10],
            'n_states': [3],
            'n_arms': [3, 4, 5],
            'transition_type': ['clinical'],
            'utility_functions': [(1, 0), (2, 4), (3, 16)],
            'thresholds': [0.2, 0.5, 0.7],
            'arm_choices': [1]
        },
    ]

    learning_episodes = 500
    n_averaging_episodes = 10
    n_iterations = 100

    save_data = True
    PATH = f'./learning-nsfinite-{learning_episodes}-{n_averaging_episodes}-{n_iterations}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    for param_sets in param_sets_list:
        print("=" * 50)
        print(f"Processing parameter set: {param_sets}") #Added to show the set being processed
        param_list = product(
            param_sets['discount_factors'],
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
            run_ns_learning_combination(params)

if __name__ == '__main__':
    main()
