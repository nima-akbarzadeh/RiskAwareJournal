import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    n_steps = [100, 1000]
    discount = 0.99999
    n_iterations = 10
    
    param_sets_list = [
        {
            'discount_factors': [discount],
            'n_steps': n_steps,
            'n_states': [3, 5],
            'n_augmnt': [10],
            'n_arms': [3, 5],
            'transition_type': ['structured'],
            'utility_functions': [(3, 4), (3, 16)],
            'thresholds': [0.5, 0.9],
            'arm_choices': [1]
        },
        {
            'discount_factors': [discount],
            'n_steps': n_steps,
            'n_states': [3],
            'n_augmnt': [5],
            'n_arms': [3, 5],
            'transition_type': ['clinical'],
            'utility_functions': [(3, 4), (3, 16)],
            'thresholds': [0.5, 0.9],
            'arm_choices': [1]
        },
    ]

    save_data = True
    PATH = f'./learning-infinite-{discount}-{n_steps}-{n_iterations}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    for param_sets in param_sets_list:
        print("=" * 50)
        print(f"Processing parameter set: {param_sets}") #Added to show the set being processed
        param_list = product(
            param_sets['discount_factors'],
            param_sets['n_steps'],
            param_sets['n_states'],
            param_sets['n_augmnt'],
            param_sets['n_arms'],
            param_sets['transition_type'],
            param_sets['utility_functions'],
            param_sets['thresholds'],
            param_sets['arm_choices'],
            [n_iterations],
            [save_data],
            [PATH]
        )
        for params in param_list:
            print('='*50)
            run_inf_learning_combination(params)

if __name__ == '__main__':
    main()
