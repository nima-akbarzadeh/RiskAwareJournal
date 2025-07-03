import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    param_sets_list = [
        # {
        #     'discount_factors': [0.99],
        #     'n_steps': [20],
        #     'n_states': [3, 4],
        #     'n_augmnt': [50],
        #     'n_arms': [5],
        #     'transition_type': ['structured'],
        #     'utility_functions': [(3, 16)],
        #     'thresholds': [0.5, 0.75],
        #     'arm_choices': [1]
        # },
        {
            'discount_factors': [0.99],
            'n_steps': [4],
            'n_states': [0],
            'n_augmnt': [10],
            'n_arms': [5],
            'transition_type': ['clinical-v2'],
            # 'transition_type': ['clinical', 'clinical-v4', 'clinical-v2', 'clinical-v3'],
            'utility_functions': [(3, 8)],
            'thresholds': [0.5],
            'arm_choices': [1]
        },
    ]

    learning_episodes = 200
    n_batches = 10
    n_iterations = 100

    save_data = True
    PATH = f'./learning-nsfinite-{learning_episodes}-{n_batches}-{n_iterations}/'
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
            [learning_episodes],  #  Add the fixed parameters to the product
            [n_batches],
            [n_iterations],
            [save_data],
            [PATH]
        )
        for params in param_list:
            print('='*50)
            run_ns_learning_combination(params)

if __name__ == '__main__':
    main()
