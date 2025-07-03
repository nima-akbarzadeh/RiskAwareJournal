import os
from utils_u import *
import warnings
warnings.filterwarnings("ignore")


def main():
    n_step = 50
    n_steps = [n_step]
    discounts = [0.99, 0.9, 0.8]
    n_iterations = 100
    ng = 50

    save_data = True
    PATH = f'./learningFinal-inf-{discounts}-{n_steps}-{n_iterations}-July1st-ng{ng}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # param_sets_list = [
    #     # {
    #     #     'discount_factors': discounts,
    #     #     'n_steps': n_steps,
    #     #     'n_states': [5],
    #     #     'n_augmnt': [ng],
    #     #     'n_discnt': n_steps,
    #     #     'n_arms': [10],
    #     #     'transition_type': ['structured'],
    #     #     'utility_functions': [(3, 8)],
    #     #     'thresholds': [0.9],
    #     #     'arm_choices': [2]
    #     # },
    #     {
    #         'discount_factors': discounts,
    #         'n_steps': n_steps,
    #         'n_states': [0],
    #         'n_augmnt': [ng],
    #         'n_discnt': n_steps,
    #         'n_arms': [5, 10],
    #         # 'transition_type': ['clinical'],
    #         'transition_type': ['clinical', 'clinical-v4', 'clinical-v3', 'clinical-v2'],
    #         'utility_functions': [(3, 4), (3, 8), (3, 16)],
    #         'thresholds': [0.5, 0.6, 0.7, 0.8, 0.9],
    #         'fraction_of_arms': [0.1, 0.3, 0.5],
    #     },
    # ]

    # for param_sets in param_sets_list:
    #     print("=" * 50)
    #     print(f"Processing parameter set: {param_sets}") #Added to show the set being processed
    #     param_list = product(
    #         param_sets['discount_factors'],
    #         param_sets['n_steps'],
    #         param_sets['n_states'],
    #         param_sets['n_augmnt'],
    #         param_sets['n_discnt'],
    #         param_sets['n_arms'],
    #         param_sets['transition_type'],
    #         param_sets['utility_functions'],
    #         param_sets['thresholds'],
    #         param_sets['fraction_of_arms'],
    #         [n_iterations],
    #         [save_data],
    #         [PATH]
    #     )
    #     for params in param_list:
    #         print('='*50)
    #         run_inf_learning_combination(params)

    # Define the constant parameters
    n_states_constant = [3, 4, 5, 10]
    transition_types_list = ['structured']
    # transition_types_list = ['clinical', 'clinical-v4', 'clinical-v3', 'clinical-v2']

    # Define your 19 base specific combinations.
    # These are hard-coded as they are your distinct, hand-picked sets.
    # Each tuple represents: (discount, n_steps, n_augmnt, n_discnt, n_arms, u_type, u_order, threshold, fraction)
    base_specific_combinations_data = [
        # (0.9, 50, 50, 50, 10, 3, 16, 0.9, 0.5),
        # (0.9, 50, 50, 50, 10, 3, 16, 0.9, 0.3),
        # (0.9, 50, 50, 50, 5, 3, 16, 0.9, 0.3),
        # (0.99, 50, 50, 50, 5, 3, 8, 0.9, 0.1),
        # (0.9, 50, 50, 50, 5, 3, 16, 0.8, 0.1),
        # (0.99, 50, 50, 50, 5, 3, 8, 0.9, 0.3),
        # (0.99, 50, 50, 50, 5, 3, 8, 0.5, 0.1),
        # (0.9, 50, 50, 50, 10, 3, 16, 0.7, 0.3),
        (0.9, 50, 50, 50, 10, 3, 8, 0.8, 0.1),
        (0.9, 50, 50, 50, 5, 3, 8, 0.7, 0.3),
        (0.9, 50, 50, 50, 10, 3, 8, 0.8, 0.3),
        # (0.99, 50, 50, 50, 10, 3, 8, 0.8, 0.5),
        # (0.9, 50, 50, 50, 5, 3, 16, 0.9, 0.5),
        # (0.99, 50, 50, 50, 5, 3, 8, 0.8, 0.3),
        # (0.99, 50, 50, 50, 10, 3, 8, 0.9, 0.5),
        (0.8, 50, 50, 50, 10, 3, 8, 0.8, 0.3),
        # (0.9, 50, 50, 50, 5, 3, 16, 0.8, 0.3),
        # (0.99, 50, 50, 50, 10, 3, 8, 0.9, 0.1),
        (0.9, 50, 50, 50, 5, 3, 8, 0.7, 0.5)
    ]

    # Process the base data into a list of dictionaries
    processed_base_combinations = []
    for combo in base_specific_combinations_data:
        processed_base_combinations.append({
            'discount_factors': combo[0],
            'n_steps': combo[1],
            'n_augmnt': combo[2],
            'n_discnt': combo[3],
            'n_arms': combo[4],
            'utility_functions': (combo[5], combo[6]), # u_type, u_order
            'thresholds': combo[7],
            'fraction_of_arms': combo[8]
        })

    # Generate the final list of parameter dictionaries
    # Each base combination is paired with each transition type
    param_sets_list = []
    for base_params in processed_base_combinations:
        for tt in transition_types_list:
            for st in n_states_constant:
                # Create a new dictionary for each specific run
                current_run_params = base_params.copy()
                current_run_params['n_states'] = st
                current_run_params['transition_type'] = tt
                param_sets_list.append(current_run_params)

    for params_dict in param_sets_list:
        print('=' * 50)
        # Print the current set of parameters being run (as a dictionary)
        print(f"Running combination: {params_dict}")

        # Unpack the dictionary values into a tuple, in the exact order expected by run_inf_learning_combination.
        # The order comes from your previous `product` call.
        params_tuple = (
            params_dict['discount_factors'],
            params_dict['n_steps'],
            params_dict['n_states'],
            params_dict['n_augmnt'],
            params_dict['n_discnt'],
            params_dict['n_arms'],
            params_dict['transition_type'],
            params_dict['utility_functions'],
            params_dict['thresholds'],
            params_dict['fraction_of_arms'],
            n_iterations,  # These are external constants, not part of the dicts in final_param_list
            save_data,
            PATH
        )

        # Call your simulation function with the prepared tuple of parameters
        run_inf_learning_combination(params_tuple)

    

if __name__ == '__main__':
    main()
