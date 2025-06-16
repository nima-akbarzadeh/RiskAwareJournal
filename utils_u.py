import numpy
import joblib
from processes import *
from whittle_v2 import *
from Markov import *
from learning import *
from utils import *
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")
RNG = numpy.random.RandomState(42)


def run_multiple_planning_combinations(param_list):
    """
    Run multiple finite horizon planning combinations in parallel and collect results.
    """
    # Determine the number of CPUs to use
    num_cpus = cpu_count() - 1
    print(f"Using {num_cpus} CPUs")
    
    # Extended evaluation keys to include all comparisons
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
    
    # Initialize result dictionaries
    results = {key: {} for key in eval_keys}
    averages = {key: {} for key in eval_keys}
    total = len(param_list)
    
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, output in enumerate(pool.imap_unordered(run_a_planning_combination, param_list), 1):
            key_value = output[0]
            metric_values = output[1:]
            
            # Store all results
            for i, value in enumerate(metric_values):
                results[eval_keys[i]][key_value] = value

            # Progress reporting with key metrics
            print(f"{count} / {total}: {key_value}")
            print(f"  --> RA vs N (obj): {metric_values[5]:.2f}%")
            print(f"  --> RU vs N (obj): {metric_values[7]:.2f}%") 
            print(f"  --> MY vs N (obj): {metric_values[8]:.2f}%")
            print(f"  --> RD vs N (obj): {metric_values[9]:.2f}%")
            print('-' * 50)
            
            # Update averages for each parameter
            param_values = key_value.split('_')
            param_names = ['nt', 'ns', 'nc', 'ut', 'th', 'fr']
            
            for param_name, param_value in zip(param_names, param_values):
                param_key = param_value
                for i, avg_key in enumerate(eval_keys):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[eval_keys[i]][key_value])
    
    return results, averages

def run_a_planning_combination(params):
    """
    Run a single finite horizon planning combination with all policy comparisons.
    """
    import time
    start_time = time.time()
    
    # Unpack parameters
    nt, ns, nc, ut, th, fr, n_iterations, save_flag, PATH = params
    key_value = f'nt{nt}_ns{ns}_nc{nc}_ut{ut}_th{th}_fr{fr}'
    
    # Derived parameters
    na = nc * ns
    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    # Initialize environment
    rew_vals = rewards(nt, na, ns)
    rew_utility_vals = rewards_utility(nt, na, ns, th, ut[0], ut[1])
    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

    # Initialize planning algorithms
    Neutral_Whittle = Whittle(ns, na, rew_vals, markov_matrix, nt)
    Neutral_Whittle.get_indices(2 * nt, nt * ns * na)

    Utility_Whittle = Whittle(ns, na, rew_utility_vals, markov_matrix, nt)
    Utility_Whittle.get_indices(2 * nt, nt * ns * na)

    RiskAware_Whittle = RiskAwareWhittle(ns, na, rew_vals, markov_matrix, nt, ut[0], ut[1], th)
    RiskAware_Whittle.get_indices(2 * nt, nt * ns * na)

    # Define all processes to evaluate
    processes = [
        ("Random", lambda *args: process_random_policy(*args)),
        ("Myopic", lambda *args: process_myopic_policy(*args)),
        ("Neutral", lambda *args: process_neutral_whittle(Neutral_Whittle, *args)),
        ("RewUtility", lambda *args: process_neutral_whittle(Utility_Whittle, *args)),
        ("RiskAware", lambda *args: process_riskaware_whittle(RiskAware_Whittle, *args))
    ]

    # Run all processes and collect results
    results = {}
    common_args = (n_iterations, nt, ns, na, nch, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1])
    
    for name, process in processes:
        rew, obj = process(*common_args)
        if save_flag:
            joblib.dump([rew, obj], f"{PATH}{key_value}_{name}.joblib")
        results[f'{name}_obj'] = numpy.mean(obj)
        results[f'{name}_rew'] = numpy.mean(rew)

    # Calculate relative improvements in objectives (percentage)
    def safe_percentage_improvement(new_val, baseline_val):
        return 100 * (new_val - baseline_val) / baseline_val if baseline_val != 0 else 0

    improve_obj_rn = safe_percentage_improvement(results['RiskAware_obj'], results['Neutral_obj'])
    improve_obj_ru = safe_percentage_improvement(results['RiskAware_obj'], results['RewUtility_obj'])
    improve_obj_un = safe_percentage_improvement(results['RewUtility_obj'], results['Neutral_obj'])
    improve_obj_mn = safe_percentage_improvement(results['Myopic_obj'], results['Neutral_obj'])
    improve_obj_dn = safe_percentage_improvement(results['Random_obj'], results['Neutral_obj'])

    # Calculate absolute differences in objectives (scaled by na)
    diff_obj_rn = na * (results['RiskAware_obj'] - results['Neutral_obj'])
    diff_obj_ru = na * (results['RiskAware_obj'] - results['RewUtility_obj'])
    diff_obj_un = na * (results['RewUtility_obj'] - results['Neutral_obj'])
    diff_obj_mn = na * (results['Myopic_obj'] - results['Neutral_obj'])
    diff_obj_dn = na * (results['Random_obj'] - results['Neutral_obj'])

    # Calculate relative improvements in rewards (from neutral perspective)
    improve_rew_nr = safe_percentage_improvement(results['Neutral_rew'], results['RiskAware_rew'])
    improve_rew_nu = safe_percentage_improvement(results['Neutral_rew'], results['RewUtility_rew'])
    improve_rew_nm = safe_percentage_improvement(results['Neutral_rew'], results['Myopic_rew'])
    improve_rew_nd = safe_percentage_improvement(results['Neutral_rew'], results['Random_rew'])

    # Calculate absolute differences in rewards (scaled by na)
    diff_rew_nr = na * (results['Neutral_rew'] - results['RiskAware_rew'])
    diff_rew_nu = na * (results['Neutral_rew'] - results['RewUtility_rew'])
    diff_rew_nm = na * (results['Neutral_rew'] - results['Myopic_rew'])
    diff_rew_nd = na * (results['Neutral_rew'] - results['Random_rew'])

    duration = time.time() - start_time
    print(f"- Duration of this round = {duration:.2f}s")

    # Return all metrics in order matching eval_keys
    return (
        key_value,
        # Objective values
        results["Neutral_obj"], results["RewUtility_obj"], results["RiskAware_obj"], 
        results["Myopic_obj"], results["Random_obj"],
        # Relative improvements in objectives
        improve_obj_rn, improve_obj_ru, improve_obj_un, improve_obj_mn, improve_obj_dn,
        # Absolute differences in objectives
        diff_obj_rn, diff_obj_ru, diff_obj_un, diff_obj_mn, diff_obj_dn,
        # Reward values
        results["Neutral_rew"], results["RewUtility_rew"], results["RiskAware_rew"],
        results["Myopic_rew"], results["Random_rew"],
        # Relative improvements in rewards
        improve_rew_nr, improve_rew_nu, improve_rew_nm, improve_rew_nd,
        # Absolute differences in rewards
        diff_rew_nr, diff_rew_nu, diff_rew_nm, diff_rew_nd
    )


def run_multiple_ns_planning_combinations(param_list):
    """
    Run multiple NS planning combinations in parallel and collect results.
    """
    # Determine the number of CPUs to use
    num_cpus = cpu_count() - 1
    print(f"Using {num_cpus} CPUs")
    
    # Extended evaluation keys to include all comparisons
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
    
    # Initialize result dictionaries
    results = {key: {} for key in eval_keys}
    averages = {key: {} for key in eval_keys}
    total = len(param_list)
    
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, output in enumerate(pool.imap_unordered(run_a_ns_planning_combination, param_list), 1):
            key_value = output[0]
            metric_values = output[1:]
            
            # Store all results
            for i, value in enumerate(metric_values):
                results[eval_keys[i]][key_value] = value

            # Progress reporting with key metrics
            print(f"{count} / {total}: {key_value}")
            print(f"  --> RA vs N (obj): {metric_values[5]:.2f}%")
            print(f"  --> RU vs N (obj): {metric_values[7]:.2f}%") 
            print(f"  --> MY vs N (obj): {metric_values[8]:.2f}%")
            print(f"  --> RD vs N (obj): {metric_values[9]:.2f}%")
            print('-' * 50)
            
            # Update averages for each parameter
            param_values = key_value.split('_')
            param_names = ['df', 'nt', 'ns', 'ng', 'nc', 'ut', 'th', 'fr']
            
            for param_name, param_value in zip(param_names, param_values):
                param_key = param_value
                for i, avg_key in enumerate(eval_keys):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[eval_keys[i]][key_value])
    
    return results, averages

def run_a_ns_planning_combination(params):
    """
    Run a single NS planning combination with all policy comparisons.
    """
    import time
    start_time = time.time()
    
    # Unpack parameters
    df, nt, ns, ng, nc, ut, th, fr, n_iterations, save_flag, PATH = params
    key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nc{nc}_ut{ut}_th{th}_fr{fr}'
    
    # Derived parameters
    na = nc * ns
    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    # Initialize environment
    rew_vals = rewards_ns(df, nt, na, ns)
    rew_utility_vals = rewards_ns_utility(df, nt, na, ns, th, ut[0], ut[1])
    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

    # Initialize planning algorithms
    Neutral_Whittle = WhittleNS(ns, na, rew_vals, markov_matrix, nt)
    Neutral_Whittle.get_indices(ng, ng * ns * na)

    Utility_Whittle = WhittleNS(ns, na, rew_utility_vals, markov_matrix, nt)
    Utility_Whittle.get_indices(ng, ng * ns * na)

    RiskAware_Whittle = RiskAwareWhittleNS([ns, ng], na, rew_vals, markov_matrix, nt, ut[0], ut[1], th)
    RiskAware_Whittle.get_indices(ng, ng * ns * na)

    # Define all processes to evaluate
    processes = [
        ("Random", lambda *args: process_ns_random_policy(*args)),
        ("Myopic", lambda *args: process_ns_myopic_policy(*args)),
        ("Neutral", lambda *args: process_ns_neutral_whittle(Neutral_Whittle, *args)),
        ("RewUtility", lambda *args: process_ns_neutral_whittle(Utility_Whittle, *args)),
        ("RiskAware", lambda *args: process_ns_riskaware_whittle(RiskAware_Whittle, *args))
    ]

    # Run all processes and collect results
    results = {}
    common_args = (n_iterations, nt, ns, na, nch, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1])
    
    for name, process in processes:
        rew, obj = process(*common_args)
        if save_flag:
            joblib.dump([rew, obj], f"{PATH}{key_value}_{name}.joblib")
        results[f'{name}_obj'] = numpy.mean(obj)
        results[f'{name}_rew'] = numpy.mean(rew)

    # Calculate relative improvements in objectives (percentage)
    def safe_percentage_improvement(new_val, baseline_val):
        return 100 * (new_val - baseline_val) / baseline_val if baseline_val != 0 else 0

    improve_obj_rn = safe_percentage_improvement(results['RiskAware_obj'], results['Neutral_obj'])
    improve_obj_ru = safe_percentage_improvement(results['RiskAware_obj'], results['RewUtility_obj'])
    improve_obj_un = safe_percentage_improvement(results['RewUtility_obj'], results['Neutral_obj'])
    improve_obj_mn = safe_percentage_improvement(results['Myopic_obj'], results['Neutral_obj'])
    improve_obj_dn = safe_percentage_improvement(results['Random_obj'], results['Neutral_obj'])

    # Calculate absolute differences in objectives (scaled by na)
    diff_obj_rn = na * (results['RiskAware_obj'] - results['Neutral_obj'])
    diff_obj_ru = na * (results['RiskAware_obj'] - results['RewUtility_obj'])
    diff_obj_un = na * (results['RewUtility_obj'] - results['Neutral_obj'])
    diff_obj_mn = na * (results['Myopic_obj'] - results['Neutral_obj'])
    diff_obj_dn = na * (results['Random_obj'] - results['Neutral_obj'])

    # Calculate relative improvements in rewards (from neutral perspective)
    improve_rew_nr = safe_percentage_improvement(results['Neutral_rew'], results['RiskAware_rew'])
    improve_rew_nu = safe_percentage_improvement(results['Neutral_rew'], results['RewUtility_rew'])
    improve_rew_nm = safe_percentage_improvement(results['Neutral_rew'], results['Myopic_rew'])
    improve_rew_nd = safe_percentage_improvement(results['Neutral_rew'], results['Random_rew'])

    # Calculate absolute differences in rewards (scaled by na)
    diff_rew_nr = na * (results['Neutral_rew'] - results['RiskAware_rew'])
    diff_rew_nu = na * (results['Neutral_rew'] - results['RewUtility_rew'])
    diff_rew_nm = na * (results['Neutral_rew'] - results['Myopic_rew'])
    diff_rew_nd = na * (results['Neutral_rew'] - results['Random_rew'])

    duration = time.time() - start_time
    print(f"- Duration of this round = {duration:.2f}s")

    # Return all metrics in order matching eval_keys
    return (
        key_value,
        # Objective values
        results["Neutral_obj"], results["RewUtility_obj"], results["RiskAware_obj"], 
        results["Myopic_obj"], results["Random_obj"],
        # Relative improvements in objectives
        improve_obj_rn, improve_obj_ru, improve_obj_un, improve_obj_mn, improve_obj_dn,
        # Absolute differences in objectives
        diff_obj_rn, diff_obj_ru, diff_obj_un, diff_obj_mn, diff_obj_dn,
        # Reward values
        results["Neutral_rew"], results["RewUtility_rew"], results["RiskAware_rew"],
        results["Myopic_rew"], results["Random_rew"],
        # Relative improvements in rewards
        improve_rew_nr, improve_rew_nu, improve_rew_nm, improve_rew_nd,
        # Absolute differences in rewards
        diff_rew_nr, diff_rew_nu, diff_rew_nm, diff_rew_nd
    )


def run_multiple_inf_planning_combinations(param_list):
    """
    Run multiple planning combinations in parallel and collect results.
    """
    # Determine the number of CPUs to use
    num_cpus = cpu_count() - 1
    print(f"Using {num_cpus} CPUs")
    
    # Extended evaluation keys to include all comparisons
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
    
    # Initialize result dictionaries
    results = {key: {} for key in eval_keys}
    averages = {key: {} for key in eval_keys}
    total = len(param_list)
    
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, output in enumerate(pool.imap_unordered(run_a_inf_planning_combination, param_list), 1):
            key_value = output[0]
            metric_values = output[1:]
            
            # Store all results
            for i, value in enumerate(metric_values):
                results[eval_keys[i]][key_value] = value

            # Progress reporting with key metrics
            print(f"{count} / {total}: {key_value}")
            print(f"  --> RA vs N (obj): {metric_values[5]:.2f}%")
            print(f"  --> RU vs N (obj): {metric_values[7]:.2f}%") 
            print(f"  --> MY vs N (obj): {metric_values[8]:.2f}%")
            print(f"  --> RD vs N (obj): {metric_values[9]:.2f}%")
            print('-' * 50)
            
            # Update averages for each parameter
            param_values = key_value.split('_')
            param_names = ['df', 'nt', 'ns', 'ng', 'nd', 'nc', 'ut', 'th', 'fr']
            
            for param_name, param_value in zip(param_names, param_values):
                param_key = param_value
                for i, avg_key in enumerate(eval_keys):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[eval_keys[i]][key_value])
    
    return results, averages

def run_a_inf_planning_combination(params):
    """
    Run a single planning combination with all policy comparisons.
    """
    import time
    start_time = time.time()
    
    # Unpack parameters
    df, nt, ns, ng, nd, nc, tt, ut, th, fr, n_iterations, save_flag, PATH = params
    key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nd{nd}_nc{nc}_tt{tt}_ut{ut}_th{th}_fr{fr}'
    
    # Derived parameters
    nd = nt  # Ensure consistency
    na = nc * ns
    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    # Initialize environment
    rew_vals = rewards_inf(df, nt, na, ns)
    rew_utility_vals = rewards_inf_utility(df, nt, na, ns, th, ut[0], ut[1])
    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    markov_matrix = get_transitions(na, ns, prob_remain, tt)

    # Initialize planning algorithms
    Neutral_Whittle = WhittleInf(ns, na, rew_vals, markov_matrix, nt, df)
    Neutral_Whittle.get_indices(ng, ng * ns * na)

    Utility_Whittle = WhittleInf(ns, na, rew_utility_vals, markov_matrix, nt, df)
    Utility_Whittle.get_indices(ng, ng * ns * na)

    RiskAware_Whittle = RiskAwareWhittleInf([ns, ng, nd], na, rew_vals, markov_matrix, df, ut[0], ut[1], th)
    RiskAware_Whittle.get_indices(ng, ng * ns * na)

    # Define all processes to evaluate
    processes = [
        ("Random", lambda *args: process_inf_random_policy(*args)),
        ("Myopic", lambda *args: process_inf_myopic_policy(*args)),
        ("Neutral", lambda *args: process_inf_neutral_whittle(Neutral_Whittle, *args)),
        ("RewUtility", lambda *args: process_inf_neutral_whittle(Utility_Whittle, *args)),
        ("RiskAware", lambda *args: process_inf_riskaware_whittle(RiskAware_Whittle, Neutral_Whittle, nd, *args))
    ]

    # Run all processes and collect results
    results = {}
    common_args = (n_iterations, df, nt, ns, na, nch, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1])
    
    for name, process in processes:
        rew, obj = process(*common_args)
        if save_flag:
            joblib.dump([rew, obj], f"{PATH}{key_value}_{name}.joblib")
        results[f'{name}_obj'] = numpy.mean(obj)
        results[f'{name}_rew'] = numpy.mean(rew)

    # Calculate relative improvements in objectives (percentage)
    def safe_percentage_improvement(new_val, baseline_val):
        return 100 * (new_val - baseline_val) / baseline_val if baseline_val != 0 else 0

    improve_obj_rn = safe_percentage_improvement(results['RiskAware_obj'], results['Neutral_obj'])
    improve_obj_ru = safe_percentage_improvement(results['RiskAware_obj'], results['RewUtility_obj'])
    improve_obj_un = safe_percentage_improvement(results['RewUtility_obj'], results['Neutral_obj'])
    improve_obj_mn = safe_percentage_improvement(results['Myopic_obj'], results['Neutral_obj'])
    improve_obj_dn = safe_percentage_improvement(results['Random_obj'], results['Neutral_obj'])

    # Calculate absolute differences in objectives (scaled by na)
    diff_obj_rn = na * (results['RiskAware_obj'] - results['Neutral_obj'])
    diff_obj_ru = na * (results['RiskAware_obj'] - results['RewUtility_obj'])
    diff_obj_un = na * (results['RewUtility_obj'] - results['Neutral_obj'])
    diff_obj_mn = na * (results['Myopic_obj'] - results['Neutral_obj'])
    diff_obj_dn = na * (results['Random_obj'] - results['Neutral_obj'])

    # Calculate relative improvements in rewards (from neutral perspective)
    improve_rew_nr = safe_percentage_improvement(results['Neutral_rew'], results['RiskAware_rew'])
    improve_rew_nu = safe_percentage_improvement(results['Neutral_rew'], results['RewUtility_rew'])
    improve_rew_nm = safe_percentage_improvement(results['Neutral_rew'], results['Myopic_rew'])
    improve_rew_nd = safe_percentage_improvement(results['Neutral_rew'], results['Random_rew'])

    # Calculate absolute differences in rewards (scaled by na)
    diff_rew_nr = na * (results['Neutral_rew'] - results['RiskAware_rew'])
    diff_rew_nu = na * (results['Neutral_rew'] - results['RewUtility_rew'])
    diff_rew_nm = na * (results['Neutral_rew'] - results['Myopic_rew'])
    diff_rew_nd = na * (results['Neutral_rew'] - results['Random_rew'])

    duration = time.time() - start_time
    print(f"- Duration of this round = {duration:.2f}s")

    # Return all metrics in order matching eval_keys
    return (
        key_value,
        # Objective values
        results["Neutral_obj"], results["RewUtility_obj"], results["RiskAware_obj"], 
        results["Myopic_obj"], results["Random_obj"],
        # Relative improvements in objectives
        improve_obj_rn, improve_obj_ru, improve_obj_un, improve_obj_mn, improve_obj_dn,
        # Absolute differences in objectives
        diff_obj_rn, diff_obj_ru, diff_obj_un, diff_obj_mn, diff_obj_dn,
        # Reward values
        results["Neutral_rew"], results["RewUtility_rew"], results["RiskAware_rew"],
        results["Myopic_rew"], results["Random_rew"],
        # Relative improvements in rewards
        improve_rew_nr, improve_rew_nu, improve_rew_nm, improve_rew_nd,
        # Absolute differences in rewards
        diff_rew_nr, diff_rew_nu, diff_rew_nm, diff_rew_nd
    )


def run_learning_combination(params):
    nt, ns, na, tt, ut, th, nc, l_episodes, n_batches, n_iterations, save_data, PATH = params

    if tt == 'structured':
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 'clinical':
        pr_ss_0 = numpy.round(numpy.linspace(0.657, 0.762, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        RNG.shuffle(pr_sp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        RNG.shuffle(pr_sp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1])
        ns=3

    key_value = f'nt{nt}_ns{ns}_na{na}_tt{tt}_ut{ut}_th{th}_nc{nc}'
    rew_vals = rewards(nt, na, ns)
    markov_matrix = get_transitions(na, ns, prob_remain, tt)
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)
    w_range = 2*nt
    w_trials = nt*ns*na

    prob_err_lr, indx_err_lr, _, obj_lr, _, obj_r = multiprocess_learn_LRAPTS(
        n_iterations, l_episodes, n_batches, nt, ns, na, nc, th, rew_vals, tt, markov_matrix, initial_states, ut[0], ut[1], 
        save_data, f'{PATH}riskaware_{key_value}.joblib', w_range, w_trials
    )
    process_and_plot(prob_err_lr, indx_err_lr, obj_r, obj_lr, 'lr', PATH, key_value)

def run_ns_learning_combination(params):
    df, nt, ns, ng, na, tt, ut, th, nc, l_episodes, n_batches, n_iterations, save_data, PATH = params

    if tt == 'structured':
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 'clinical':
        pr_ss_0 = numpy.round(numpy.linspace(0.657, 0.762, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        RNG.shuffle(pr_sp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        RNG.shuffle(pr_sp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1])
        ns=3
    elif tt == 'clinical-v2':
        pr_ss_0 = numpy.round(numpy.linspace(0.596, 0.690, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sr_0 = numpy.round(numpy.linspace(0.045, 0.061, na), 3)
        RNG.shuffle(pr_sr_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        RNG.shuffle(pr_sp_0)
        pr_rr_0 = numpy.round(numpy.linspace(0.759, 0.822, na), 3)
        RNG.shuffle(pr_rr_0)
        pr_rp_0 = numpy.round(numpy.linspace(0.130, 0.169, na), 3)
        RNG.shuffle(pr_rp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.733, 0.801, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sr_1 = numpy.round(numpy.linspace(0.047, 0.078, na), 3)
        RNG.shuffle(pr_sr_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        RNG.shuffle(pr_sp_1)
        pr_rr_1 = numpy.round(numpy.linspace(0.758, 0.847, na), 3)
        RNG.shuffle(pr_rr_1)
        pr_rp_1 = numpy.round(numpy.linspace(0.121, 0.193, na), 3)
        RNG.shuffle(pr_rp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sr_0, pr_sp_0, pr_rr_0, pr_rp_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_sp_1, pr_rr_1, pr_rp_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v3':
        pr_ss_0 = numpy.round(numpy.linspace(0.668, 0.738, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sr_0 = numpy.round(numpy.linspace(0.045, 0.061, na), 3)
        RNG.shuffle(pr_sr_0)
        pr_rr_0 = numpy.round(numpy.linspace(0.831, 0.870, na), 3)
        RNG.shuffle(pr_rr_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.782, 0.833, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sr_1 = numpy.round(numpy.linspace(0.047, 0.078, na), 3)
        RNG.shuffle(pr_sr_1)
        pr_rr_1 = numpy.round(numpy.linspace(0.807, 0.879, na), 3)
        RNG.shuffle(pr_rr_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sr_0, pr_rr_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_rr_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v4':
        pr_ss_0 = numpy.round(numpy.linspace(0.713, 0.799, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.829, 0.885, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_pp_0, pr_ss_1, pr_pp_1])
        ns = 3


    key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_na{na}_tt{tt}_ut{ut}_th{th}_nc{nc}'
    rew_vals = rewards_ns(df, nt, na, ns)
    markov_matrix = get_transitions(na, ns, prob_remain, tt)
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)
    w_range = 2*nt
    w_trials = nt*ns*na

    prob_err_lr, indx_err_lr, _, obj_lr, _, obj_r = multiprocess_ns_learn_LRAPTS(
        n_iterations, l_episodes, n_batches, nt, ns, ng, na, nc, th, rew_vals, tt, markov_matrix, initial_states, ut[0], ut[1], 
        save_data, f'{PATH}ns_riskaware_{key_value}.joblib', w_range, w_trials
    )
    process_and_plot(prob_err_lr, indx_err_lr, obj_r, obj_lr, 'lr', PATH, key_value)

def run_inf_learning_combination(params):
    df, nt, ns, ng, nd, na, tt, ut, th, nc, n_iterations, save_data, PATH = params

    if tt == 'structured':
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 'clinical':
        pr_ss_0 = numpy.round(numpy.linspace(0.657, 0.762, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        RNG.shuffle(pr_sp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        RNG.shuffle(pr_sp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1])
        ns=3
    elif tt == 'clinical-v2':
        pr_ss_0 = numpy.round(numpy.linspace(0.596, 0.690, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sr_0 = numpy.round(numpy.linspace(0.045, 0.061, na), 3)
        RNG.shuffle(pr_sr_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        RNG.shuffle(pr_sp_0)
        pr_rr_0 = numpy.round(numpy.linspace(0.759, 0.822, na), 3)
        RNG.shuffle(pr_rr_0)
        pr_rp_0 = numpy.round(numpy.linspace(0.130, 0.169, na), 3)
        RNG.shuffle(pr_rp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.733, 0.801, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sr_1 = numpy.round(numpy.linspace(0.047, 0.078, na), 3)
        RNG.shuffle(pr_sr_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        RNG.shuffle(pr_sp_1)
        pr_rr_1 = numpy.round(numpy.linspace(0.758, 0.847, na), 3)
        RNG.shuffle(pr_rr_1)
        pr_rp_1 = numpy.round(numpy.linspace(0.121, 0.193, na), 3)
        RNG.shuffle(pr_rp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sr_0, pr_sp_0, pr_rr_0, pr_rp_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_sp_1, pr_rr_1, pr_rp_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v3':
        pr_ss_0 = numpy.round(numpy.linspace(0.668, 0.738, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sr_0 = numpy.round(numpy.linspace(0.045, 0.061, na), 3)
        RNG.shuffle(pr_sr_0)
        pr_rr_0 = numpy.round(numpy.linspace(0.831, 0.870, na), 3)
        RNG.shuffle(pr_rr_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.782, 0.833, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sr_1 = numpy.round(numpy.linspace(0.047, 0.078, na), 3)
        RNG.shuffle(pr_sr_1)
        pr_rr_1 = numpy.round(numpy.linspace(0.807, 0.879, na), 3)
        RNG.shuffle(pr_rr_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sr_0, pr_rr_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_rr_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v4':
        pr_ss_0 = numpy.round(numpy.linspace(0.713, 0.799, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.829, 0.885, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_pp_0, pr_ss_1, pr_pp_1])
        ns = 3

    key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nd{nd}_na{na}_tt{tt}_ut{ut}_th{th}_nc{nc}'
    rew_vals = rewards_inf(df, nt, na, ns)
    markov_matrix = get_transitions(na, ns, prob_remain, tt)
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)
    w_range = 2*ng
    w_trials = ng*ns*na

    oracle_results, riskaware_results, neutral_results, baseline_results = multiprocess_inf_learn_LRAPTSDE(
        n_iterations, df, nt, ns, ng, nd, na, nc, th, rew_vals, tt, markov_matrix, initial_states, ut[0], ut[1], 
        save_data, f'{PATH}inf_riskaware_{key_value}.joblib', w_range, w_trials
    )
    # REGRET FOR OBJECTIVES
    process_and_plot_inf(
        prob_err=riskaware_results["transitionerrors"], 
        indx_err=riskaware_results["indexerrors"], 
        perf_ref=oracle_results["objectives"], 
        perf_lrn=riskaware_results["objectives"], 
        perf_bas=neutral_results["objectives"], 
        perf_stt = {key: value for key, value in baseline_results.items() if key.endswith('_obj') and not key.startswith('RAP')}, 
        suffix='lr', 
        path=PATH, 
        key_value=key_value,
    )

def run_avg_learning_combination(params):
    nt, ns, na, tt, nc, n_iterations, save_data, PATH = params

    if tt == 'structured':
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 'clinical':
        pr_ss_0 = numpy.round(numpy.linspace(0.657, 0.762, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        RNG.shuffle(pr_sp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        RNG.shuffle(pr_sp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1])
        ns=3
    elif tt == 'clinical-v2':
        pr_ss_0 = numpy.round(numpy.linspace(0.596, 0.690, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sr_0 = numpy.round(numpy.linspace(0.045, 0.061, na), 3)
        RNG.shuffle(pr_sr_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        RNG.shuffle(pr_sp_0)
        pr_rr_0 = numpy.round(numpy.linspace(0.759, 0.822, na), 3)
        RNG.shuffle(pr_rr_0)
        pr_rp_0 = numpy.round(numpy.linspace(0.130, 0.169, na), 3)
        RNG.shuffle(pr_rp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.733, 0.801, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sr_1 = numpy.round(numpy.linspace(0.047, 0.078, na), 3)
        RNG.shuffle(pr_sr_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        RNG.shuffle(pr_sp_1)
        pr_rr_1 = numpy.round(numpy.linspace(0.758, 0.847, na), 3)
        RNG.shuffle(pr_rr_1)
        pr_rp_1 = numpy.round(numpy.linspace(0.121, 0.193, na), 3)
        RNG.shuffle(pr_rp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sr_0, pr_sp_0, pr_rr_0, pr_rp_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_sp_1, pr_rr_1, pr_rp_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v3':
        pr_ss_0 = numpy.round(numpy.linspace(0.668, 0.738, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_sr_0 = numpy.round(numpy.linspace(0.045, 0.061, na), 3)
        RNG.shuffle(pr_sr_0)
        pr_rr_0 = numpy.round(numpy.linspace(0.831, 0.870, na), 3)
        RNG.shuffle(pr_rr_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.782, 0.833, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_sr_1 = numpy.round(numpy.linspace(0.047, 0.078, na), 3)
        RNG.shuffle(pr_sr_1)
        pr_rr_1 = numpy.round(numpy.linspace(0.807, 0.879, na), 3)
        RNG.shuffle(pr_rr_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sr_0, pr_rr_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_rr_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v4':
        pr_ss_0 = numpy.round(numpy.linspace(0.713, 0.799, na), 3)
        RNG.shuffle(pr_ss_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        RNG.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.829, 0.885, na), 3)
        RNG.shuffle(pr_ss_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        RNG.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_pp_0, pr_ss_1, pr_pp_1])
        ns = 3

    key_value = f'nt{nt}_ns{ns}_na{na}_tt{tt}_nc{nc}'
    rew_vals = rewards(nt, na, ns)
    markov_matrix = get_transitions(na, ns, prob_remain, tt)
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)
    w_range = nt
    w_trials = nt*ns*na

    prob_err_lr, indx_err_lr, rew_lr, rew_n = multiprocess_avg_learn_TSDE(
        n_iterations, nt, ns, na, nc, rew_vals, tt, markov_matrix, initial_states, 
        save_data, f'{PATH}avg_{key_value}.joblib', w_range, w_trials
    )
    process_and_plot(prob_err_lr, indx_err_lr, rew_n, rew_lr, 'ln', PATH, key_value)

 