import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    n_steps = [1000]
    discounts = [0.9, 0.99]
    n_iterations = 10

    param_sets_list = [
        {
            'discount_factors': discounts,
            'n_steps': n_steps,
            'n_states': [3],
            'n_augmnt': [50],
            'n_discnt': [100],
            'n_arms': [2, 4],
            'transition_type': ['structured'],
            'utility_functions': [(2, 8)],
            'thresholds': [0.2, 0.5],
            'arm_choices': [1]
        },
        {
            'discount_factors': discounts,
            'n_steps': n_steps,
            'n_states': [3],
            'n_augmnt': [5],
            'n_discnt': [100],
            'n_arms': [2, 4],
            'transition_type': ['clinical'],
            'utility_functions': [(2, 8)],
            'thresholds': [0.2, 0.5],
            'arm_choices': [1]
        },
    ]

    save_data = True
    PATH = f'./learning-infinite-{discounts}-{n_steps}-{n_iterations}/'
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
            param_sets['n_discnt'],
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
