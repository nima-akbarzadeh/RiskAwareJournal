import os
import numpy
import glob
import joblib
import pandas as pd
from utils_u import run_multiple_ns_planning_combinations
import warnings
warnings.filterwarnings("ignore")


PATH = './planning-nsfinite-May25-T20/'
# PATH = f'./planning-nsfinite-May25-dis9999/'
if not os.path.exists(PATH):
    os.makedirs(PATH)

def main():

    # Combinations
    param_sets = {
        'discount_factors': [0.67, 0.75, 0.8, 0.9],
        'n_steps': [20],
        'n_states': [2, 3, 4],
        'n_augmnt': [10],
        'n_arms_coefficient': [3, 4, 5],
        'utility_functions': [(1, 0), (2, 4), (2, 8), (2, 16), (3, 4), (3, 8), (3, 16)],
        'thresholds': [numpy.round(0.1 * n, 1) for n in range(1, 10)],
        'fraction_of_arms': [0.1, 0.3, 0.5]
    }
    # param_sets = {
    #     'discount_factors': [0.9999],
    #     'n_steps': [3, 4, 5],
    #     'n_states': [2, 3, 4],
    #     'n_augmnt': [10],
    #     'n_arms_coefficient': [3, 4, 5],
    #     'utility_functions': [(1, 0), (2, 4), (2, 8), (2, 16), (3, 4), (3, 8), (3, 16)],
    #     'thresholds': [numpy.round(0.1 * n, 1) for n in range(1, 10)],
    #     'fraction_of_arms': [0.1, 0.3, 0.5]
    # }

    # Iterations
    n_iterations = 100

    # Saving the results
    save_flag = True
    
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
    import time
    start_time = time.time()
    results, averages = run_multiple_ns_planning_combinations(param_list)
    print(f'Duration: {time.time() - start_time}')

    # Save results to Excel
    df1 = pd.DataFrame({f'MEAN-{key.capitalize()}': value for key, value in results.items()})
    df1.index.name = 'Key'
    df1.to_excel(f'{PATH}res_ns.xlsx')

    df2 = pd.DataFrame({f'MEAN-{key.capitalize()}': {k: numpy.mean(v) if v else 0 for k, v in avg.items()} for key, avg in averages.items()})
    df2.index.name = 'Key'
    df2.to_excel(f'{PATH}resavg_ns.xlsx')

def combine_joblib_files(input_dir, output_filename=PATH+"all_files.joblib"):
    """
    Combines multiple .joblib files from a specified directory into a single .joblib file.

    Args:
        input_dir (str): The path to the directory containing the .joblib files.
        output_filename (str, optional): The name of the output .joblib file.
            Defaults to "all_files.joblib".
    """
    # Use glob to find all .joblib files in the specified directory
    file_list = glob.glob(os.path.join(input_dir, "*.joblib"))

    if not file_list:
        print(f"No .joblib files found in the directory: {input_dir}")
        return

    # Load the data from each file
    data = {}
    for file_name in file_list:
        try:
            loaded_data = joblib.load(file_name)
            key = os.path.splitext(os.path.basename(file_name))[0]
            data[key] = loaded_data
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            print(f"Skipping {file_name}...")
            continue

    # Save the combined data to a new .joblib file
    try:
        joblib.dump(data, output_filename, compress=3)
        print(f"Successfully combined files into {output_filename}")
    except Exception as e:
        print(f"Error saving combined file: {e}")

def split_joblib_file(input_filename):
    """
    Splits a single .joblib file (created by combine_joblib_files) into multiple individual .joblib files.
    The keys in the loaded data are used as the filenames.

    Args:
        input_filename (str): The full path to the input .joblib file.
    """
    try:
        # Load the combined data
        combined_data = joblib.load(input_filename)
    except Exception as e:
        print(f"Error loading combined file {input_filename}: {e}")
        return

    print(f"Splitting file {input_filename} into individual files:")
    # Iterate through the data and save each item as a separate .joblib file
    for file_name, file_data in combined_data.items():
        try:
            # Construct the full filename with the .joblib extension
            output_file_name = f"{file_name}.joblib"
            joblib.dump(file_data, output_file_name, compress=3)
            print(f"Created file: {output_file_name}")
        except Exception as e:
            print(f"Error creating file {file_name}.joblib: {e}")
            continue


if __name__ == '__main__':
    main()

    # 1. Combine files
    combine_joblib_files(PATH)

    # # 2. Split the combined file
    # combined_file_path = PATH + "all_files.joblib"
    # split_joblib_file(combined_file_path)
    