import os
import numpy
import pandas as pd
from utils_u import *
import warnings
warnings.filterwarnings("ignore")

def main():

    # Combinations
    param_sets = {
        'discount_factors': [0.9, 0.95],
        'n_steps': [100],
        'n_augmnt': [10],
        'n_states': [5, 4, 3, 2],
        'n_arms_coefficient': [5, 4, 3],
        'utility_functions': [(1, 0), (2, 4), (2, 8), (2, 16), (3, 4), (3, 8), (3, 16)],
        'thresholds': [np.round(0.1 * n, 1) for n in range(1, 10)],
        'fraction_of_arms': [0.1, 0.3, 0.5]
    }

    # Iterations
    n_iterations = 100

    # Saving the results
    save_flag = True
    PATH = f'./planning-infinite/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Main run
    param_list = [
        (df, nt, ns, ng, nc, ut, th, fr, n_iterations, save_flag, PATH)
        for df in param_sets['discount_factors']
        for nt in param_sets['n_steps']
        for ns in param_sets['n_states']
        for ng in param_sets['n_augmnt']
        for nc in param_sets['n_arms_coefficient']
        for ut in param_sets['utility_functions']
        for th in param_sets['thresholds']
        for fr in param_sets['fraction_of_arms']
    ]
    results, averages = run_multiple_inf_planning_combinations(param_list)

    # Save results to Excel
    df1 = pd.DataFrame({f'MEAN-{key.capitalize()}': value for key, value in results.items()})
    df1.index.name = 'Key'
    df1.to_excel(f'{PATH}res_inf.xlsx')

    df2 = pd.DataFrame({f'MEAN-{key.capitalize()}': {k: numpy.mean(v) if v else 0 for k, v in avg.items()} for key, avg in averages.items()})
    df2.index.name = 'Key'
    df2.to_excel(f'{PATH}resavg_inf.xlsx')


if __name__ == '__main__':
    main()