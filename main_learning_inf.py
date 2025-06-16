import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    n_steps = [100]
    discounts = [0.99, 0.9, 0.8]
    n_iterations = 20

    param_sets_list = [
        # {
        #     'discount_factors': discounts,
        #     'n_steps': n_steps,
        #     'n_states': [5],
        #     'n_augmnt': [20],
        #     'n_discnt': [20],
        #     'n_arms': [2],
        #     'transition_type': ['structured'],
        #     'utility_functions': [(3, 16), (3, 4), (3, 8)],
        #     'thresholds': [0.5],
        #     'arm_choices': [1]
        # },
        {
            'discount_factors': discounts,
            'n_steps': n_steps,
            'n_states': [0],
            'n_augmnt': [10],
            'n_discnt': [50],
            'n_arms': [5, 10],
            'transition_type': ['clinical', 'clinical-v4', 'clinical-v3', 'clinical-v2'],
            'utility_functions': [(3, 16), (3, 4), (3, 8)],
            'thresholds': [0.25, 0.5, 0.75],
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
