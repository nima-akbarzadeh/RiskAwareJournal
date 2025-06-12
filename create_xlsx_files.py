import os
import re
import joblib
import numpy
import pandas as pd
from collections import defaultdict
import ast  # To safely evaluate the tuple string for 'ut'
from tqdm import tqdm  # Import tqdm

# --- Configuration ---
# IMPORTANT: Set this to the actual path where ALL your .joblib files are saved
PATH = './FINAL/planning-infinite-June25/'  # Example: '/path/to/output/data/'
# --- End Configuration ---

param_filters = {
    'th': [0.6, 0.7],
    'fr': [0.5],
}

# Define the keys for evaluation metrics, updated to include all policies
eval_keys = [
    # Objective values
    'Neutral_Obj', 'RewUtility_Obj', 'RiskAware_Obj', 'Myopic_Obj', 'Random_Obj',
    # Relative improvements in objectives
    'RI_Obj_RiskAware_to_Neutral', 'RI_Obj_RiskAware_to_RewUtility', 'RI_Obj_RewUtility_to_Neutral',
    'RI_Obj_Myopic_to_Neutral', 'RI_Obj_Random_to_Neutral',
    # Absolute differences in objectives  
    'DF_Obj_RiskAware_to_Neutral', 'DF_Obj_RiskAware_to_RewUtility', 'DF_Obj_RewUtility_to_Neutral',
    'DF_Obj_Myopic_to_Neutral', 'DF_Obj_Random_to_Neutral',
    # Reward values
    'Neutral_Rew', 'RewUtility_Rew', 'RiskAware_Rew', 'Myopic_Rew', 'Random_Rew',
    # Relative improvements in rewards (from perspective of neutral being better)
    'RI_Rew_Neutral_to_RiskAware', 'RI_Rew_Neutral_to_RewUtility', 'RI_Rew_Neutral_to_Myopic', 'RI_Rew_Neutral_to_Random',
    # Absolute differences in rewards
    'DF_Rew_Neutral_to_RiskAware', 'DF_Rew_Neutral_to_RewUtility', 'DF_Rew_Neutral_to_Myopic', 'DF_Rew_Neutral_to_Random'
]

# Dictionary to hold the loaded raw data, separated by experiment type
loaded_data = {
    'finite': defaultdict(dict),
    'nonstationary': defaultdict(dict),
    'infinite': defaultdict(dict)
}

# Updated regex patterns to handle both old and new formats
# New infinite pattern (with 'tt' parameter)
infinite_new_pattern = re.compile(
    r"df([\d.]+)_nt(\d+)_ns(\d+)_ng(\d+)_nd(\d+)_nc(\d+)_tt([^_]+)_ut\((.*?)\)_th([\d.]+)_fr([\d.]+)"
    r"_(Neutral|RewUtility|RiskAware|Myopic|Random)"
    r"\.joblib$"
)
# Old infinite pattern (without 'tt' parameter)
infinite_old_pattern = re.compile(
    r"df([\d.]+)_nt(\d+)_ns(\d+)_ng(\d+)_nd(\d+)_nc(\d+)_ut\((.*?)\)_th([\d.]+)_fr([\d.]+)"
    r"_(Neutral|RewUtility|RiskAware|Myopic|Random)"
    r"\.joblib$"
)
# Nonstationary pattern (no 'nd' parameter)
nonstationary_pattern = re.compile(
    r"df([\d.]+)_nt(\d+)_ns(\d+)_ng(\d+)_nc(\d+)_ut\((.*?)\)_th([\d.]+)_fr([\d.]+)"
    r"_(Neutral|RewUtility|RiskAware|Myopic|Random)"
    r"\.joblib$"
)
# Finite pattern (no 'df' parameter)
finite_pattern = re.compile(
    r"nt(\d+)_ns(\d+)_nc(\d+)_ut\((.*?)\)_th([\d.]+)_fr([\d.]+)"
    r"_(Neutral|RewUtility|RiskAware|Myopic|Random)"
    r"\.joblib$"
)

def parse_filename(filename):
    """
    Parse filename and return experiment type, parameters, and process name.
    Returns tuple: (experiment_type, params, key_value, process_name, na)
    """
    
    # Try new infinite pattern first (with 'tt')
    match = infinite_new_pattern.match(filename)
    if match:
        try:
            df = float(match.group(1))
            nt = int(match.group(2))
            ns = int(match.group(3))
            ng = int(match.group(4))
            nd = int(match.group(5))
            nc = int(match.group(6))
            tt = match.group(7)  # Keep 'tt' as string (e.g., "structured")
            ut_str = match.group(8)
            ut = ast.literal_eval(f"({ut_str})")
            th = float(match.group(9))
            fr = float(match.group(10))
            process_name = match.group(11)

            params = {'df': df, 'nt': nt, 'ns': ns, 'ng': ng, 'nd': nd, 'nc': nc, 'tt': tt, 'ut': ut, 'th': th, 'fr': fr}
            key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nd{nd}_nc{nc}_tt{tt}_ut{ut}_th{th}_fr{fr}'
            na = nc * ns
            
            return ('infinite', params, key_value, process_name, na)
        except Exception as e:
            tqdm.write(f"Warning: Error parsing New Infinite params from {filename}. Error: {e}")
    
    # Try old infinite pattern (without 'tt')
    match = infinite_old_pattern.match(filename)
    if match:
        try:
            df = float(match.group(1))
            nt = int(match.group(2))
            ns = int(match.group(3))
            ng = int(match.group(4))
            nd = int(match.group(5))
            nc = int(match.group(6))
            ut_str = match.group(7)
            ut = ast.literal_eval(f"({ut_str})")
            th = float(match.group(8))
            fr = float(match.group(9))
            process_name = match.group(10)

            params = {'df': df, 'nt': nt, 'ns': ns, 'ng': ng, 'nd': nd, 'nc': nc, 'ut': ut, 'th': th, 'fr': fr}
            key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nd{nd}_nc{nc}_ut{ut}_th{th}_fr{fr}'
            na = nc * ns
            
            return ('infinite', params, key_value, process_name, na)
        except Exception as e:
            tqdm.write(f"Warning: Error parsing Old Infinite params from {filename}. Error: {e}")

    # Try nonstationary pattern (has 'df' but no 'nd')
    match = nonstationary_pattern.match(filename)
    if match:
        try:
            df = float(match.group(1))
            nt = int(match.group(2))
            ns = int(match.group(3))
            ng = int(match.group(4))
            nc = int(match.group(5))
            ut_str = match.group(6)
            ut = ast.literal_eval(f"({ut_str})")
            th = float(match.group(7))
            fr = float(match.group(8))
            process_name = match.group(9)

            params = {'df': df, 'nt': nt, 'ns': ns, 'ng': ng, 'nc': nc, 'ut': ut, 'th': th, 'fr': fr}
            key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nc{nc}_ut{ut}_th{th}_fr{fr}'
            na = nc * ns
            
            return ('nonstationary', params, key_value, process_name, na)
        except Exception as e:
            tqdm.write(f"Warning: Error parsing Nonstationary params from {filename}. Error: {e}")

    # Try finite pattern (no 'df')
    match = finite_pattern.match(filename)
    if match:
        try:
            nt = int(match.group(1))
            ns = int(match.group(2))
            nc = int(match.group(3))
            ut_str = match.group(4)
            ut = ast.literal_eval(f"({ut_str})")
            th = float(match.group(5))
            fr = float(match.group(6))
            process_name = match.group(7)

            params = {'nt': nt, 'ns': ns, 'nc': nc, 'ut': ut, 'th': th, 'fr': fr}
            key_value = f'nt{nt}_ns{ns}_nc{nc}_ut{ut}_th{th}_fr{fr}'
            na = nc * ns
            
            return ('finite', params, key_value, process_name, na)
        except Exception as e:
            tqdm.write(f"Warning: Error parsing Finite params from {filename}. Error: {e}")
    
    return (None, None, None, None, None)

print(f"Scanning directory: {PATH}...")
print("Classifying files based on parameters:")
print("- Finite: No 'df' key in filename.")
print("- Nonstationary: Has 'df' key but NO 'nd' key.")
print("- Infinite: Has 'df' key AND 'nd' key (supports both old and new formats with/without 'tt').")

# Get list of relevant files first for tqdm
try:
    all_files = os.listdir(PATH)
    joblib_files = [f for f in all_files if f.endswith(".joblib")]
    if not joblib_files:
        print("\nWarning: No '.joblib' files found in the specified directory.")
        exit()  # Exit if no files to process
    print(f"\nFound {len(joblib_files)} '.joblib' files to process.")
except FileNotFoundError:
    print(f"\nError: Directory not found: {PATH}")
    exit()  # Exit if directory doesn't exist

count_loaded = 0
count_skipped = 0
file_counts = {'finite': 0, 'nonstationary': 0, 'infinite': 0}
format_counts = {'infinite_new': 0, 'infinite_old': 0}

# 1. Load data from all joblib files and classify them
# Wrap the joblib_files list with tqdm for the progress bar
for filename in tqdm(joblib_files, desc="Scanning files", unit="file", ncols=100):
    
    # Parse filename using the new unified function
    experiment_type, params, key_value, process_name, na = parse_filename(filename)
    
    # Load data if parsing was successful
    if experiment_type and key_value and process_name is not None:

        ## MODIFIED: Check if the file should be filtered based on its parameters.
        should_skip = False
        for param_key, rejected_values in param_filters.items():
            # Check if the parameter from the filter exists in the parsed parameters
            if param_key in params and params[param_key] in rejected_values:
                tqdm.write(f"Info: Skipping '{filename}' due to filter rule: {param_key}={params[param_key]}")
                should_skip = True
                break  # A match was found, no need to check other filters for this file

        if should_skip:
            count_skipped += 1
            continue # Skip to the next file

        try:
            filepath = os.path.join(PATH, filename)
            rew_array, obj_array = joblib.load(filepath)

            obj_mean = numpy.mean(obj_array)
            rew_mean = numpy.mean(rew_array)

            loaded_data[experiment_type][key_value][process_name] = {
                'obj': obj_mean,
                'rew': rew_mean,
                'na': na,
                'params': params
            }
            
            file_counts[experiment_type] += 1
            
            # Track format types for infinite horizon
            if experiment_type == 'infinite':
                if 'tt' in params:
                    format_counts['infinite_new'] += 1
                else:
                    format_counts['infinite_old'] += 1
                    
            count_loaded += 1
        except FileNotFoundError:
            tqdm.write(f"Warning: File not found (might have been moved/deleted): {filepath}")
            count_skipped += 1
        except Exception as e:
            tqdm.write(f"Warning: Could not load data from {filepath}. Error: {e}")
            count_skipped += 1
    else:  # File was .joblib but didn't match expected patterns or failed parsing
        tqdm.write(f"Warning: Filename format mismatch, skipped: {filename}")
        count_skipped += 1

print(f"\nFinished loading data.")  # Print after the loop/progress bar finishes
print(f"Successfully loaded data from {count_loaded} files.")
print(f"  Finite: {file_counts['finite']} files")
print(f"  Non-Stationary: {file_counts['nonstationary']} files")
print(f"  Infinite Horizon: {file_counts['infinite']} files")
if file_counts['infinite'] > 0:
    print(f"    - New format (with tt): {format_counts['infinite_new']} files")
    print(f"    - Old format (without tt): {format_counts['infinite_old']} files")
if count_skipped > 0:
    print(f"Skipped {count_skipped} files due to errors or format mismatch.")

# Safe percentage improvement function (same as in planning functions)
def safe_percentage_improvement(new_val, baseline_val):
    return 100 * (new_val - baseline_val) / baseline_val if baseline_val != 0 else 0

# --- Process data and save results for each experiment type ---
for exp_type in ['finite', 'nonstationary', 'infinite']:
    print(f"\nProcessing loaded data for type: {exp_type.upper()}...")

    if not loaded_data[exp_type]:
        print(f"No data loaded for type '{exp_type}'. Skipping.")
        continue

    # Dictionaries for results and averages for this type
    results = {key: {} for key in eval_keys}
    averages = {key: defaultdict(list) for key in eval_keys}
    processed_keys = 0

    # 2. Process each key_value combination within the type
    for key_value, process_data in loaded_data[exp_type].items():
        # Check if all required policies are present
        required_policies = ["Neutral", "RewUtility", "RiskAware", "Myopic", "Random"]
        if all(policy in process_data for policy in required_policies):
            # Extract results for all policies
            n_res = process_data["Neutral"]
            ru_res = process_data["RewUtility"]
            ra_res = process_data["RiskAware"]
            my_res = process_data["Myopic"]
            rd_res = process_data["Random"]
            
            na = n_res['na']
            params = n_res['params']

            # Extract objective and reward values
            neutral_obj = n_res['obj']
            rewutility_obj = ru_res['obj']
            riskaware_obj = ra_res['obj']
            myopic_obj = my_res['obj']
            random_obj = rd_res['obj']
            
            neutral_rew = n_res['rew']
            rewutility_rew = ru_res['rew']
            riskaware_rew = ra_res['rew']
            myopic_rew = my_res['rew']
            random_rew = rd_res['rew']

            # Calculate relative improvements in objectives (percentage)
            improve_obj_rn = safe_percentage_improvement(riskaware_obj, neutral_obj)
            improve_obj_ru = safe_percentage_improvement(riskaware_obj, rewutility_obj)
            improve_obj_un = safe_percentage_improvement(rewutility_obj, neutral_obj)
            improve_obj_mn = safe_percentage_improvement(myopic_obj, neutral_obj)
            improve_obj_dn = safe_percentage_improvement(random_obj, neutral_obj)

            # Calculate absolute differences in objectives (scaled by na)
            diff_obj_rn = na * (riskaware_obj - neutral_obj)
            diff_obj_ru = na * (riskaware_obj - rewutility_obj)
            diff_obj_un = na * (rewutility_obj - neutral_obj)
            diff_obj_mn = na * (myopic_obj - neutral_obj)
            diff_obj_dn = na * (random_obj - neutral_obj)

            # Calculate relative improvements in rewards (from neutral perspective)
            improve_rew_nr = safe_percentage_improvement(neutral_rew, riskaware_rew)
            improve_rew_nu = safe_percentage_improvement(neutral_rew, rewutility_rew)
            improve_rew_nm = safe_percentage_improvement(neutral_rew, myopic_rew)
            improve_rew_nd = safe_percentage_improvement(neutral_rew, random_rew)

            # Calculate absolute differences in rewards (scaled by na)
            diff_rew_nr = na * (neutral_rew - riskaware_rew)
            diff_rew_nu = na * (neutral_rew - rewutility_rew)
            diff_rew_nm = na * (neutral_rew - myopic_rew)
            diff_rew_nd = na * (neutral_rew - random_rew)

            # All calculated values in order matching eval_keys
            calculated_values = [
                # Objective values
                neutral_obj, rewutility_obj, riskaware_obj, myopic_obj, random_obj,
                # Relative improvements in objectives
                improve_obj_rn, improve_obj_ru, improve_obj_un, improve_obj_mn, improve_obj_dn,
                # Absolute differences in objectives
                diff_obj_rn, diff_obj_ru, diff_obj_un, diff_obj_mn, diff_obj_dn,
                # Reward values
                neutral_rew, rewutility_rew, riskaware_rew, myopic_rew, random_rew,
                # Relative improvements in rewards
                improve_rew_nr, improve_rew_nu, improve_rew_nm, improve_rew_nd,
                # Absolute differences in rewards
                diff_rew_nr, diff_rew_nu, diff_rew_nm, diff_rew_nd
            ]

            # Store in results dict
            for i, value in enumerate(calculated_values):
                results[eval_keys[i]][key_value] = value

            # Update averages dict
            param_map = params.copy()
            param_map['ut'] = str(param_map['ut'])

            for param_name, param_val in param_map.items():
                param_key = f"{param_name}_{param_val}"
                for i, avg_key in enumerate(eval_keys):
                    averages[avg_key][param_key].append(calculated_values[i])

            processed_keys += 1
        else:
            missing_policies = [policy for policy in required_policies if policy not in process_data]
            print(f"  Warning: Skipped key '{key_value}' for type '{exp_type}' due to missing policies: {missing_policies}")

    print(f"Finished processing {processed_keys} parameter combinations for type '{exp_type}'.")

    # 3. Calculate final averages and additional statistics
    print(f"Calculating final averages and additional statistics for type '{exp_type}'...")
    final_averages = {
        key: {param_key: numpy.mean(values) if values else 0 for param_key, values in avg_data.items()}
        for key, avg_data in averages.items()
    }
    
    # Calculate additional statistics for RI_Obj_RiskAware_to_Neutral
    target_metric = 'RI_Obj_RiskAware_to_Neutral'
    if target_metric in averages:
        # Initialize dictionaries for additional statistics
        max_stats = {param_key: numpy.max(values) if values else 0 for param_key, values in averages[target_metric].items()}
        min_stats = {param_key: numpy.min(values) if values else 0 for param_key, values in averages[target_metric].items()}
        below_zero_stats = {param_key: numpy.mean([v < 0 for v in values]) if values else 0 for param_key, values in averages[target_metric].items()}
        
        # Add these to final_averages with appropriate names
        final_averages[f'{target_metric}_MAX'] = max_stats
        final_averages[f'{target_metric}_MIN'] = min_stats
        final_averages[f'{target_metric}_BELOW_ZERO'] = below_zero_stats

    # --- Start: Add natural sorting logic ---
    unique_param_keys = set()
    for avg_data in averages.values():
        unique_param_keys.update(avg_data.keys())

    def natural_sort_key(s):
        parts = re.split(r'(\d+\.?\d*|\d+)', s)
        key = []
        for part in parts:
            if not part:
                continue
            try:
                key.append(float(part))
            except ValueError:
                key.append(part.lower())
        return key

    sorted_param_keys = sorted(list(unique_param_keys), key=natural_sort_key)
    # --- End: Add natural sorting logic ---

    # 4. Save results to Excel
    suffix = ""
    if exp_type == 'nonstationary':
        suffix = "_ns"
    elif exp_type == 'infinite':
        suffix = "_inf"

    results_excel_path = os.path.join(PATH, f'res{suffix}_loaded.xlsx')
    averages_excel_path = os.path.join(PATH, f'resavg{suffix}_loaded.xlsx')

    print(f"Saving detailed results ({exp_type}) to: {results_excel_path}")
    try:
        df_results = pd.DataFrame(results)
        df_results.columns = [f'MEAN-{col.replace("_", " ").title().replace(" ", "")}' for col in df_results.columns]
        df_results.index.name = 'Key'
        df_results.to_excel(results_excel_path)
        print(f"Detailed results ({exp_type}) saved successfully.")
    except Exception as e:
        print(f"Error saving detailed results ({exp_type}) to Excel: {e}")

    # --- Saving averaged results (df_averages) with sorting and additional statistics ---
    print(f"Saving averaged results ({exp_type}) to: {averages_excel_path}")
    try:
        df_averages = pd.DataFrame(final_averages)
        valid_sorted_keys = [key for key in sorted_param_keys if key in df_averages.index]
        df_averages = df_averages.reindex(index=valid_sorted_keys)

        # Apply column name transformation
        new_columns = []
        for col in df_averages.columns:
            if col.endswith('_MAX'):
                base_col = col.replace('_MAX', '')
                new_col_name = f'MAX-{base_col.replace("_", " ").title().replace(" ", "")}'
            elif col.endswith('_MIN'):
                base_col = col.replace('_MIN', '')
                new_col_name = f'MIN-{base_col.replace("_", " ").title().replace(" ", "")}'
            elif col.endswith('_BELOW_ZERO'):
                base_col = col.replace('_BELOW_ZERO', '')
                new_col_name = f'BELOW_ZERO-{base_col.replace("_", " ").title().replace(" ", "")}'
            else:
                new_col_name = f'MEAN-{col.replace("_", " ").title().replace(" ", "")}'
            new_columns.append(new_col_name)
        
        df_averages.columns = new_columns
        df_averages.index.name = 'Param_Value'

        df_averages.to_excel(averages_excel_path)
        print(f"Averaged results ({exp_type}) saved successfully.")
    except Exception as e:
        print(f"Error saving averaged results ({exp_type}) to Excel: {e}")

print("\nScript finished.")
