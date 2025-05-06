import os
import re
import joblib
import numpy
import pandas as pd
from collections import defaultdict
import ast # To safely evaluate the tuple string for 'ut'
from tqdm import tqdm # Import tqdm

# --- Configuration ---
# IMPORTANT: Set this to the actual path where ALL your .joblib files are saved
PATH = './planning-finite-May25/' # Example: '/path/to/output/data/'
# --- End Configuration ---

# Define the keys for evaluation metrics, same across all types
eval_keys = [
    'Neutral_Obj', 'RewUtility_Obj', 'RiskAware_Obj',
    'RI_Obj_RiskAware_to_Neutral', 'RI_Obj_RiskAware_to_RewUtility', 'RI_Obj_RewUtility_to_Neutral',
    'DF_Obj_RiskAware_to_Neutral', 'DF_Obj_RiskAware_to_RewUtility', 'DF_Obj_RewUtility_to_Neutral',
    'Neutral_Rew', 'RewUtility_Rew', 'RiskAware_Rew',
    'RI_Rew_Neutral_to_RiskAware', 'RI_Rew_Neutral_to_RewUtility',
    'DF_Rew_Neutral_to_RiskAware', 'DF_Rew_Neutral_to_RewUtility'
]

# Dictionary to hold the loaded raw data, separated by experiment type
loaded_data = {
    'finite': defaultdict(dict),
    'nonstationary': defaultdict(dict),
    'infinite': defaultdict(dict)
}

# Regex patterns for different filename structures
df_present_pattern = re.compile(
    r"df([\d.]+)_nt(\d+)_ns(\d+)_ng(\d+)_nc(\d+)_ut\((.*?)\)_th([\d.]+)_fr([\d.]+)"
    r"_(Neutral|RewUtility|RiskAware)"
    r"\.joblib$"
)
finite_pattern = re.compile(
    r"nt(\d+)_ns(\d+)_nc(\d+)_ut\((.*?)\)_th([\d.]+)_fr([\d.]+)"
    r"_(Neutral|RewUtility|RiskAware)"
    r"\.joblib$"
)

print(f"Scanning directory: {PATH}...")
print("Classifying files based on parameters:")
print("- Finite: No 'df' key in filename.")
print("- Nonstationary: Has 'df' key AND 'nt' <= 100.")
print("- Infinite: Has 'df' key AND 'nt' > 100.")

# Get list of relevant files first for tqdm
try:
    all_files = os.listdir(PATH)
    joblib_files = [f for f in all_files if f.endswith(".joblib")]
    if not joblib_files:
        print("\nWarning: No '.joblib' files found in the specified directory.")
        exit() # Exit if no files to process
    print(f"\nFound {len(joblib_files)} '.joblib' files to process.")
except FileNotFoundError:
    print(f"\nError: Directory not found: {PATH}")
    exit() # Exit if directory doesn't exist


count_loaded = 0
count_skipped = 0
file_counts = {'finite': 0, 'nonstationary': 0, 'infinite': 0}

# 1. Load data from all joblib files and classify them
# Wrap the joblib_files list with tqdm for the progress bar
for filename in tqdm(joblib_files, desc="Scanning files", unit="file", ncols=100): # Added tqdm wrapper

    matched = False
    experiment_type = None
    params = {}
    key_value = None
    process_name = None
    na = None

    # First, try matching the pattern WITH 'df'
    match_df = df_present_pattern.match(filename)
    if match_df:
        matched = True
        try:
            df = float(match_df.group(1))
            nt = int(match_df.group(2)) # Extract nt to classify
            ns = int(match_df.group(3))
            ng = int(match_df.group(4))
            nc = int(match_df.group(5))
            ut_str = match_df.group(6)
            ut = ast.literal_eval(f"({ut_str})")
            th = float(match_df.group(7))
            fr = float(match_df.group(8))
            process_name = match_df.group(9)

            params = {'df': df, 'nt': nt, 'ns': ns, 'ng': ng, 'nc': nc, 'ut': ut, 'th': th, 'fr': fr}
            key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nc{nc}_ut{ut}_th{th}_fr{fr}'
            na = nc * ns

            # Classify based on 'nt' value
            if nt > 100:
                 experiment_type = 'infinite'
            else: # nt <= 100
                 experiment_type = 'nonstationary'

        except Exception as e:
            # Use tqdm.write for messages inside the loop to avoid interfering with the bar
            tqdm.write(f"Warning: Error parsing DF-Present params from {filename}. Error: {e}")
            matched = False

    # If not matched above, try matching the pattern WITHOUT 'df' (Finite)
    if not matched:
        match_finite = finite_pattern.match(filename)
        if match_finite:
            matched = True
            experiment_type = 'finite' # Classified as Finite
            try:
                nt = int(match_finite.group(1))
                ns = int(match_finite.group(2))
                nc = int(match_finite.group(3))
                ut_str = match_finite.group(4)
                ut = ast.literal_eval(f"({ut_str})")
                th = float(match_finite.group(5))
                fr = float(match_finite.group(6))
                process_name = match_finite.group(7)

                params = {'nt': nt, 'ns': ns, 'nc': nc, 'ut': ut, 'th': th, 'fr': fr}
                key_value = f'nt{nt}_ns{ns}_nc{nc}_ut{ut}_th{th}_fr{fr}'
                na = nc * ns

            except Exception as e:
                tqdm.write(f"Warning: Error parsing Finite params from {filename}. Error: {e}")
                matched = False

    # Load data if a pattern was matched and parsed successfully
    if matched and experiment_type and key_value and process_name is not None:
        try:
            filepath = os.path.join(PATH, filename)
            rew_array, obj_array = joblib.load(filepath)

            loaded_data[experiment_type][key_value][process_name] = {
                'obj': numpy.mean(obj_array),
                'rew': numpy.mean(rew_array),
                'na': na,
                'params': params
            }
            file_counts[experiment_type] += 1
            count_loaded += 1
        except FileNotFoundError:
            tqdm.write(f"Warning: File not found (might have been moved/deleted): {filepath}")
            count_skipped += 1
        except Exception as e:
            tqdm.write(f"Warning: Could not load data from {filepath}. Error: {e}")
            count_skipped += 1
    else: # File was .joblib but didn't match expected patterns or failed parsing
        if matched == False: # Avoid double-warning if parsing failed above
            tqdm.write(f"Warning: Filename format mismatch, skipped: {filename}")
        count_skipped += 1


# The rest of the script remains the same...

print(f"\nFinished loading data.") # Print after the loop/progress bar finishes
print(f"Successfully loaded data from {count_loaded} files.")
print(f"  Finite: {file_counts['finite']} files")
print(f"  Non-Stationary: {file_counts['nonstationary']} files")
print(f"  Infinite Horizon: {file_counts['infinite']} files")
if count_skipped > 0:
    print(f"Skipped {count_skipped} files due to errors or format mismatch.")


# --- Process data and save results for each experiment type ---
for exp_type in ['finite', 'nonstationary', 'infinite']:
    print(f"\nProcessing loaded data for type: {exp_type.upper()}...")
    # ...(Processing and saving logic remains unchanged)...

    if not loaded_data[exp_type]:
        print(f"No data loaded for type '{exp_type}'. Skipping.")
        continue

    # Dictionaries for results and averages for this type
    results = {key: {} for key in eval_keys}
    averages = {key: defaultdict(list) for key in eval_keys}
    processed_keys = 0

    # 2. Process each key_value combination within the type
    # You could add a tqdm wrapper here too if processing is slow
    # for key_value, process_data in tqdm(loaded_data[exp_type].items(), desc=f"Processing {exp_type}", unit="key"):
    for key_value, process_data in loaded_data[exp_type].items():
        if "Neutral" in process_data and "RewUtility" in process_data and "RiskAware" in process_data:
            n_res = process_data["Neutral"]
            ru_res = process_data["RewUtility"]
            ra_res = process_data["RiskAware"]
            na = n_res['na']
            params = n_res['params']

            # Calculate derived metrics (logic is identical)
            neutral_obj = n_res['obj']
            rewutility_obj = ru_res['obj']
            riskaware_obj = ra_res['obj']
            neutral_rew = n_res['rew']
            rewutility_rew = ru_res['rew']
            riskaware_rew = ra_res['rew']

            improve_obj_rn = 100 * (riskaware_obj - neutral_obj) / neutral_obj if neutral_obj != 0 else 0
            improve_obj_ru = 100 * (riskaware_obj - rewutility_obj) / rewutility_obj if rewutility_obj != 0 else 0
            improve_obj_un = 100 * (rewutility_obj - neutral_obj) / neutral_obj if neutral_obj != 0 else 0

            diff_obj_rn = na * (riskaware_obj - neutral_obj)
            diff_obj_ru = na * (riskaware_obj - rewutility_obj)
            diff_obj_un = na * (rewutility_obj - neutral_obj)

            improve_rew_nr = 100 * (neutral_rew - riskaware_rew) / riskaware_rew if riskaware_rew != 0 else 0
            improve_rew_nu = 100 * (neutral_rew - rewutility_rew) / rewutility_rew if rewutility_rew != 0 else 0

            diff_rew_nr = na * (neutral_rew - riskaware_rew)
            diff_rew_nu = na * (neutral_rew - rewutility_rew)

            calculated_values = [
                neutral_obj, rewutility_obj, riskaware_obj,
                improve_obj_rn, improve_obj_ru, improve_obj_un,
                diff_obj_rn, diff_obj_ru, diff_obj_un,
                neutral_rew, rewutility_rew, riskaware_rew,
                improve_rew_nr, improve_rew_nu,
                diff_rew_nr, diff_rew_nu
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
             # Use tqdm.write if you add a progress bar to this loop later
             print(f"  Warning: Skipped key '{key_value}' for type '{exp_type}' due to missing process results (Neutral/RewUtility/RiskAware).")


    print(f"Finished processing {processed_keys} parameter combinations for type '{exp_type}'.")

    # 3. Calculate final averages
    print(f"Calculating final averages for type '{exp_type}'...")
    final_averages = {
        key: {param_key: numpy.mean(values) if values else 0 for param_key, values in avg_data.items()}
        for key, avg_data in averages.items()
    }

    # --- Start: Add natural sorting logic ---

    # Get all unique parameter keys from the original averages data before final calculation
    # This ensures we capture all keys even if some lists were empty
    unique_param_keys = set()
    for avg_data in averages.values(): # 'averages' is the dict with lists of values
        unique_param_keys.update(avg_data.keys())

    # Define the natural sort key function
    def natural_sort_key(s):
        """
        Create a sort key for natural sorting (e.g., 'item2' before 'item10').
        Splits string into text and number parts. Handles floats/decimals too.
        """
        # Split based on digits OR digits with a decimal point
        parts = re.split(r'(\d+\.?\d*|\d+)', s) 
        key = []
        for part in parts:
            if not part: # Skip empty strings from split
                continue
            # Try converting numerical parts to float for proper comparison
            try: 
                key.append(float(part))
            except ValueError: # Keep text parts as lowercase strings
                key.append(part.lower())
        return key

    # Sort the unique keys naturally
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

    # --- Saving averaged results (df_averages) with sorting ---
    print(f"Saving averaged results ({exp_type}) to: {averages_excel_path}")
    try:
        # Create DataFrame for averaged results using the pre-calculated final_averages
        df_averages = pd.DataFrame(final_averages)

        # Reindex the DataFrame using the naturally sorted keys
        # This ensures the rows appear in the desired order
        # Make sure the DataFrame actually contains the keys before reindexing
        # Filter sorted_param_keys to only those present in the df index
        valid_sorted_keys = [key for key in sorted_param_keys if key in df_averages.index]
        df_averages = df_averages.reindex(index=valid_sorted_keys) # Use only valid keys

        # Set column names and index name
        df_averages.columns = [f'MEAN-{col.replace("_", " ").title().replace(" ", "")}' for col in df_averages.columns]
        df_averages.index.name = 'Param_Value' # Index now represents sorted parameter keys

        # Save to Excel
        df_averages.to_excel(averages_excel_path)
        print(f"Averaged results ({exp_type}) saved successfully.")
    except Exception as e:
        print(f"Error saving averaged results ({exp_type}) to Excel: {e}")


print("\nScript finished.")
