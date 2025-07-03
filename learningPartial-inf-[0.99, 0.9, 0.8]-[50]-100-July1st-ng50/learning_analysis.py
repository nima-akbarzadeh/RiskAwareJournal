# -*- coding: utf-8 -*-
"""
This script analyzes the results of reinforcement learning simulations.

Based on the latest user instruction, this version has been modified to
exclude 'n_states' from the set of parameters that define a unique
experimental combination. It assumes 'n_states' is an internal property
of the 'transition_type' models.

This will result in a final report with 54 unique parameter rows,
aggregating results from different 'n_states' into a single row.
"""
import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

def calculate_performance(results_dict):
    """
    Calculates a single performance score from a results dictionary.
    """
    if "objectives" not in results_dict or results_dict["objectives"].size == 0:
        return np.nan

    sum_over_arms = np.sum(results_dict["objectives"], axis=2)
    mean_over_iterations = np.mean(sum_over_arms, axis=0)
    last_time_step_value = mean_over_iterations[-1]
    
    return last_time_step_value

def get_performance_category(o_perf, ra_perf, n_perf):
    """
    Classifies the outcome into one of seven mutually exclusive categories.
    """
    if ra_perf > o_perf and o_perf >= n_perf:
        return "RA > O >= N"
    elif n_perf > o_perf and o_perf >= ra_perf:
        return "N > O >= RA"
    elif n_perf > ra_perf and ra_perf > o_perf:
        return "N > RA > O"
    elif o_perf >= ra_perf and ra_perf >= n_perf:
        return "O >= RA >= N"
    elif o_perf >= n_perf and n_perf > ra_perf:
        return "O >= N > RA"
    elif ra_perf >= n_perf and n_perf > o_perf:
        return "RA >= N > O"
    else:
        return "Other"

def parse_filename_and_params(filepath):
    """
    Parses a filepath to extract simulation parameters using regex.
    """
    filename = os.path.basename(filepath)
    
    pattern = re.compile(
        r"df([\d.]+)_nt(\d+)_ns(\d+)_ng(\d+)_nd(\d+)_na(\d+)_tt([\w-]+)_ut\(([\d, ]+)\)_th([\d.]+)_fr(\d.)"
    )
    match = pattern.search(filename)

    if not match:
        return None

    groups = match.groups()
    
    try:
        u_tuple_parts = groups[7].split(',')
        u_type = int(u_tuple_parts[0].strip())
        u_order = int(u_tuple_parts[1].strip())

        params = {
            'discount': float(groups[0]),
            'n_steps': int(groups[1]),
            'n_states': int(groups[2]), # Still parsed, just not used as a key
            'n_augmnt': int(groups[3]),
            'n_discnt': int(groups[4]),
            'n_arms': int(groups[5]),
            'transition_type': groups[6],
            'u_type': u_type,
            'u_order': u_order,
            'threshold': float(groups[8]),
            'n_choices': int(groups[9]),
        }
        return params
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse parts of filename '{filename}'. Error: {e}")
        return None


def analyze_simulation_results():
    """
    Main function to find, process, and analyze all simulation joblib files.
    """
    folder_name = 'learningPartial-inf-[0.99, 0.9, 0.8]-[50]-100-July1st-ng50'
    escaped_path = glob.escape(folder_name)
    search_pattern = os.path.join(escaped_path, '*.joblib')
    joblib_files = glob.glob(search_pattern)

    # search_path = './'
    # joblib_files = glob.glob(f'{search_path}/**/*.joblib', recursive=True)

    # search_path = './'
    # joblib_files = glob.glob(f'{search_path}/**/*.joblib', recursive=True)

    if not joblib_files:
        print("Fatal Error: No .joblib files found.")
        return

    print(f"Found {len(joblib_files)} total .joblib files to process...")
    
    all_runs_data = []
    
    for f_path in joblib_files:
        try:
            params = parse_filename_and_params(f_path)
            if params is None:
                print(f"Warning: Could not parse parameters from filename, skipping: {os.path.basename(f_path)}")
                continue

            oracle_res, riskaware_res, neutral_res, _ = joblib.load(f_path)
            
            o_perf = calculate_performance(oracle_res)
            ra_perf = calculate_performance(riskaware_res)
            n_perf = calculate_performance(neutral_res)
            
            if any(np.isnan([o_perf, ra_perf, n_perf])):
                print(f"Warning: Missing or invalid 'objectives' data in file, skipping: {os.path.basename(f_path)}")
                continue

            category = get_performance_category(o_perf, ra_perf, n_perf)
            
            run_data = params.copy()
            run_data['category'] = category
            all_runs_data.append(run_data)

        except Exception as e:
            print(f"Error: Failed to process file {os.path.basename(f_path)}. Reason: {e}")

    if not all_runs_data:
        print("\nNo valid data was processed. Cannot generate CSV report.")
        return
        
    df = pd.DataFrame(all_runs_data)
    
    # --- KEY CHANGE HERE ---
    # Define the columns for a unique row. 'n_states' has been REMOVED from this list
    # as per the instruction to aggregate results across different n_states values.
    parameter_cols = [
        'discount', 'n_steps', 'n_augmnt', 'n_discnt', 'n_arms',
        'u_type', 'u_order', 'threshold', 'n_choices'
    ]
    
    pivot_df = df.pivot_table(
        index=parameter_cols,
        columns='category',
        values='transition_type',
        aggfunc=lambda x: ', '.join(sorted(x))
    ).fillna('')

    category_order = [
        "O >= RA >= N", "RA > O >= N", "O >= N > RA",
        "RA >= N > O", "N > O >= RA", "N > RA > O", "Other"
    ]
    for cat in category_order:
        if cat not in pivot_df.columns:
            pivot_df[cat] = ''
    pivot_df = pivot_df[category_order]

    # Use a new filename to avoid confusion with the previous 108-row version
    output_filename = 'transition_type_analysis.csv'
    pivot_df.to_csv(output_filename)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Processed {len(all_runs_data)} valid simulation files.")
    print(f"Detailed report saved to: {output_filename}")
    print(f"The report correctly contains {len(pivot_df)} unique parameter combinations (rows).")
    print("\nNOTE: Results from different 'n_states' have been aggregated into single rows.")


if __name__ == '__main__':
    analyze_simulation_results()
    