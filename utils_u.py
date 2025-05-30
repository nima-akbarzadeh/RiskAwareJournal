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


def run_multiple_planning_combinations(param_list):

    # Determine the number of CPUs to use
    num_cpus = cpu_count()-1
    print(f"Using {num_cpus} CPUs")
    
    eval_keys = ['Neutral_Obj', 'RewUtility_Obj', 'RiskAware_Obj',
                 'RI_Obj_RiskAware_to_Neutral', 'RI_Obj_RiskAware_to_RewUtility', 'RI_Obj_RewUtility_to_Neutral',
                 'DF_Obj_RiskAware_to_Neutral', 'DF_Obj_RiskAware_to_RewUtility', 'DF_Obj_RewUtility_to_Neutral',
                 'Neutral_Rew', 'RewUtility_Rew', 'RiskAware_Rew', 
                 'RI_Rew_Neutral_to_RiskAware', 'RI_Rew_Neutral_to_RewUtility',
                 'DF_Rew_Neutral_to_RiskAware', 'DF_Rew_Neutral_to_RewUtility']
    results = {key: {} for key in eval_keys}
    averages = {key: {} for key in eval_keys}
    total = len(param_list)
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, output in enumerate(pool.imap_unordered(run_a_planning_combination, param_list), 1):
            key_value, raavg_n, raavg_ru, raavg_ra, improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, travg_n, travg_ru, travg_ra, improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu = output
            for i, value in enumerate([raavg_n, raavg_ru, raavg_ra, improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, travg_n, travg_ru, travg_ra, improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu]):
                results[eval_keys[i]][key_value] = value

            print(f"{count} / {total}: {key_value} ---> MEAN-Rel-RN: {improve_obj_rn}, MEAN-Rel-UN: {improve_obj_un}")
            for _, value in zip(['nt', 'ns', 'nc', 'ut', 'th', 'fr'], output[0].split('_')):
                param_key = f'{value}'
                for i, avg_key in enumerate(eval_keys):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[eval_keys[i]][key_value])
    
    return results, averages


def run_a_planning_combination(params):
    nt, ns, nc, ut, th, fr, n_iterations, save_flag, PATH = params
    key_value = f'nt{nt}_ns{ns}_nc{nc}_ut{ut}_th{th}_fr{fr}'
    na = nc * ns

    rew_vals = rewards(nt, na, ns)
    rew_utility_vals = rewards_utility(nt, na, ns, th, ut[0], ut[1])
    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

    Neutral_Whittle = Whittle(ns, na, rew_vals, markov_matrix, nt)
    Neutral_Whittle.get_indices(2*nt, nt*ns*na)

    Utility_Whittle = Whittle(ns, na, rew_utility_vals, markov_matrix, nt)
    Utility_Whittle.get_indices(2*nt, nt*ns*na)

    RiskAware_Whittle = RiskAwareWhittle(ns, na, rew_vals, markov_matrix, nt, ut[0], ut[1], th)
    RiskAware_Whittle.get_indices(2*nt, nt*ns*na)

    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    processes = [
        ("Neutral", lambda *args: process_neutral_whittle(Neutral_Whittle, *args)),
        ("RewUtility", lambda *args: process_neutral_whittle(Utility_Whittle, *args)),
        ("RiskAware", lambda *args: process_riskaware_whittle(RiskAware_Whittle, *args))
    ]

    results = {}
    for name, process in processes:
        rew, obj = process(n_iterations, nt, ns, na, nch, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1])
        if save_flag:
            joblib.dump([rew, obj], f"{PATH}{key_value}_{name}.joblib")
        results[name+'_obj'] = numpy.mean(obj)
        results[name+'_rew'] = numpy.mean(rew)

    improve_obj_rn = 100 * (results['RiskAware_obj'] - results['Neutral_obj']) / results['Neutral_obj'] if results['Neutral_obj'] != 0 else 0
    improve_obj_ru = 100 * (results['RiskAware_obj'] - results['RewUtility_obj']) / results['RewUtility_obj'] if results['RewUtility_obj'] != 0 else 0
    improve_obj_un = 100 * (results['RewUtility_obj'] - results['Neutral_obj']) / results['Neutral_obj'] if results['Neutral_obj'] != 0 else 0

    diff_obj_rn = na * (results['RiskAware_obj'] - results['Neutral_obj'])
    diff_obj_ru = na * (results['RiskAware_obj'] - results['RewUtility_obj'])
    diff_obj_un = na * (results['RewUtility_obj'] - results['Neutral_obj'])

    improve_rew_nr = 100 * (results['Neutral_rew'] - results['RiskAware_rew']) / results['RiskAware_rew'] if results['RiskAware_rew'] != 0 else 0
    improve_rew_nu = 100 * (results['Neutral_rew'] - results['RewUtility_rew']) / results['RewUtility_rew'] if results['RewUtility_rew'] != 0 else 0

    diff_rew_nr = na * (results['Neutral_rew'] - results['RiskAware_rew'])
    diff_rew_nu = na * (results['Neutral_rew'] - results['RewUtility_rew'])

    return key_value, results["Neutral_obj"], results["RewUtility_obj"], results["RiskAware_obj"], improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, results["Neutral_rew"], results["RewUtility_rew"], results["RiskAware_rew"], improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu


def run_multiple_ns_planning_combinations(param_list):

    # Determine the number of CPUs to use
    num_cpus = cpu_count()-1
    print(f"Using {num_cpus} CPUs")
    
    eval_keys = ['Neutral_Obj', 'RewUtility_Obj', 'RiskAware_Obj',
                 'RI_Obj_RiskAware_to_Neutral', 'RI_Obj_RiskAware_to_RewUtility', 'RI_Obj_RewUtility_to_Neutral',
                 'DF_Obj_RiskAware_to_Neutral', 'DF_Obj_RiskAware_to_RewUtility', 'DF_Obj_RewUtility_to_Neutral',
                 'Neutral_Rew', 'RewUtility_Rew', 'RiskAware_Rew', 
                 'RI_Rew_Neutral_to_RiskAware', 'RI_Rew_Neutral_to_RewUtility',
                 'DF_Rew_Neutral_to_RiskAware', 'DF_Rew_Neutral_to_RewUtility']
    results = {key: {} for key in eval_keys}
    averages = {key: {} for key in eval_keys}
    total = len(param_list)
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, output in enumerate(pool.imap_unordered(run_a_ns_planning_combination, param_list), 1):
            key_value, raavg_n, raavg_ru, raavg_ra, improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, travg_n, travg_ru, travg_ra, improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu = output
            for i, value in enumerate([raavg_n, raavg_ru, raavg_ra, improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, travg_n, travg_ru, travg_ra, improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu]):
                results[eval_keys[i]][key_value] = value

            print(f"{count} / {total}: {key_value} ---> MEAN-Rel-RN: {improve_obj_rn}")
            for _, value in zip(['df', 'nt', 'ns', 'ng', 'nc', 'ut', 'th', 'fr'], output[0].split('_')):
                param_key = f'{value}'
                for i, avg_key in enumerate(eval_keys):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[eval_keys[i]][key_value])
    
    return results, averages


def run_a_ns_planning_combination(params):
    import time
    start_time = time.time()
    df, nt, ns, ng, nc, ut, th, fr, n_iterations, save_flag, PATH = params
    key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nc{nc}_ut{ut}_th{th}_fr{fr}'
    na = nc * ns

    rew_vals = rewards_ns(df, nt, na, ns)
    rew_utility_vals = rewards_ns_utility(df, nt, na, ns, th, ut[0], ut[1])
    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

    Neutral_Whittle = WhittleNS(ns, na, rew_vals, markov_matrix, nt)
    Neutral_Whittle.get_indices(ng, ng*ns*na)

    Utility_Whittle = WhittleNS(ns, na, rew_utility_vals, markov_matrix, nt)
    Utility_Whittle.get_indices(ng, ng*ns*na)

    RiskAware_Whittle = RiskAwareWhittleNS([ns, ng], na, rew_vals, markov_matrix, nt, ut[0], ut[1], th)
    RiskAware_Whittle.get_indices(ng, ng*ns*na)

    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    processes = [
        ("Neutral", lambda *args: process_ns_neutral_whittle(Neutral_Whittle, *args)),
        ("RewUtility", lambda *args: process_ns_neutral_whittle(Utility_Whittle, *args)),
        ("RiskAware", lambda *args: process_ns_riskaware_whittle(RiskAware_Whittle, *args))
    ]

    results = {}
    for name, process in processes:
        rew, obj = process(n_iterations, nt, ns, na, nch, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1])
        if save_flag:
            joblib.dump([rew, obj], f"{PATH}{key_value}_{name}.joblib")
        results[name+'_obj'] = numpy.mean(obj)
        results[name+'_rew'] = numpy.mean(rew)

    improve_obj_rn = 100 * (results['RiskAware_obj'] - results['Neutral_obj']) / results['Neutral_obj'] if results['Neutral_obj'] != 0 else 0
    improve_obj_ru = 100 * (results['RiskAware_obj'] - results['RewUtility_obj']) / results['RewUtility_obj'] if results['RewUtility_obj'] != 0 else 0
    improve_obj_un = 100 * (results['RewUtility_obj'] - results['Neutral_obj']) / results['Neutral_obj'] if results['Neutral_obj'] != 0 else 0

    diff_obj_rn = na * (results['RiskAware_obj'] - results['Neutral_obj'])
    diff_obj_ru = na * (results['RiskAware_obj'] - results['RewUtility_obj'])
    diff_obj_un = na * (results['RewUtility_obj'] - results['Neutral_obj'])

    improve_rew_nr = 100 * (results['Neutral_rew'] - results['RiskAware_rew']) / results['RiskAware_rew'] if results['RiskAware_rew'] != 0 else 0
    improve_rew_nu = 100 * (results['Neutral_rew'] - results['RewUtility_rew']) / results['RewUtility_rew'] if results['RewUtility_rew'] != 0 else 0

    diff_rew_nr = na * (results['Neutral_rew'] - results['RiskAware_rew'])
    diff_rew_nu = na * (results['Neutral_rew'] - results['RewUtility_rew'])

    print(f"- Duration of this round = {time.time() - start_time}")

    return key_value, results["Neutral_obj"], results["RewUtility_obj"], results["RiskAware_obj"], improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, results["Neutral_rew"], results["RewUtility_rew"], results["RiskAware_rew"], improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu


def run_multiple_inf_planning_combinations(param_list):

    # Determine the number of CPUs to use
    num_cpus = cpu_count()-1
    print(f"Using {num_cpus} CPUs")
    
    eval_keys = ['Neutral_Obj', 'RewUtility_Obj', 'RiskAware_Obj',
                 'RI_Obj_RiskAware_to_Neutral', 'RI_Obj_RiskAware_to_RewUtility', 'RI_Obj_RewUtility_to_Neutral',
                 'DF_Obj_RiskAware_to_Neutral', 'DF_Obj_RiskAware_to_RewUtility', 'DF_Obj_RewUtility_to_Neutral',
                 'Neutral_Rew', 'RewUtility_Rew', 'RiskAware_Rew', 
                 'RI_Rew_Neutral_to_RiskAware', 'RI_Rew_Neutral_to_RewUtility',
                 'DF_Rew_Neutral_to_RiskAware', 'DF_Rew_Neutral_to_RewUtility']
    results = {key: {} for key in eval_keys}
    averages = {key: {} for key in eval_keys}
    total = len(param_list)
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, output in enumerate(pool.imap_unordered(run_a_inf_planning_combination, param_list), 1):
            key_value, raavg_n, raavg_ru, raavg_ra, improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, travg_n, travg_ru, travg_ra, improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu = output
            for i, value in enumerate([raavg_n, raavg_ru, raavg_ra, improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, travg_n, travg_ru, travg_ra, improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu]):
                results[eval_keys[i]][key_value] = value

            print(f"{count} / {total}: {key_value} ---> MEAN-Rel-RN: {improve_obj_rn}")
            print(f"{count} / {total}: {key_value} ---> MEAN-Rel-UN: {improve_obj_un}")
            print('-'*20)
            for _, value in zip(['df', 'nt', 'ns', 'ng', 'nd', 'nc', 'ut', 'th', 'fr'], output[0].split('_')):
                param_key = f'{value}'
                for i, avg_key in enumerate(eval_keys):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[eval_keys[i]][key_value])
    
    return results, averages


def run_a_inf_planning_combination(params):

    import time
    start_time = time.time()
    df, nt, ns, ng, nd, nc, ut, th, fr, n_iterations, save_flag, PATH = params
    key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nd{nd}_nc{nc}_ut{ut}_th{th}_fr{fr}'
    na = nc * ns

    rew_vals = rewards_inf(df, nt, na, ns)
    rew_utility_vals = rewards_inf_utility(df, nt, na, ns, th, ut[0], ut[1])
    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    markov_matrix = get_transitions(na, ns, prob_remain, 'structured')

    Neutral_Whittle = WhittleInf(ns, na, rew_vals, markov_matrix, nt, df)
    Neutral_Whittle.get_indices(ng, ng*ns*na)

    Utility_Whittle = WhittleInf(ns, na, rew_utility_vals, markov_matrix, nt, df)
    Utility_Whittle.get_indices(ng, ng*ns*na)

    RiskAware_Whittle = RiskAwareWhittleInf([ns, ng, nd], na, rew_vals, markov_matrix, df, ut[0], ut[1], th)
    RiskAware_Whittle.get_indices(ng, ng*ns*na)

    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    processes = [
        ("Neutral", lambda *args: process_inf_neutral_whittle(Neutral_Whittle, *args)),
        ("RewUtility", lambda *args: process_inf_neutral_whittle(Utility_Whittle, *args)),
        ("RiskAware", lambda *args: process_inf_riskaware_whittle(RiskAware_Whittle, Neutral_Whittle, nd, *args))
    ]

    results = {}
    for name, process in processes:
        rew, obj = process(n_iterations, df, nt, ns, na, nch, th, rew_vals, markov_matrix, initial_states, ut[0], ut[1])
        if save_flag:
            joblib.dump([rew, obj], f"{PATH}{key_value}_{name}.joblib")
        results[name+'_obj'] = numpy.mean(obj)
        results[name+'_rew'] = numpy.mean(rew)

    improve_obj_rn = 100 * (results['RiskAware_obj'] - results['Neutral_obj']) / results['Neutral_obj'] if results['Neutral_obj'] != 0 else 0
    improve_obj_ru = 100 * (results['RiskAware_obj'] - results['RewUtility_obj']) / results['RewUtility_obj'] if results['RewUtility_obj'] != 0 else 0
    improve_obj_un = 100 * (results['RewUtility_obj'] - results['Neutral_obj']) / results['Neutral_obj'] if results['Neutral_obj'] != 0 else 0

    diff_obj_rn = na * (results['RiskAware_obj'] - results['Neutral_obj'])
    diff_obj_ru = na * (results['RiskAware_obj'] - results['RewUtility_obj'])
    diff_obj_un = na * (results['RewUtility_obj'] - results['Neutral_obj'])

    improve_rew_nr = 100 * (results['Neutral_rew'] - results['RiskAware_rew']) / results['RiskAware_rew'] if results['RiskAware_rew'] != 0 else 0
    improve_rew_nu = 100 * (results['Neutral_rew'] - results['RewUtility_rew']) / results['RewUtility_rew'] if results['RewUtility_rew'] != 0 else 0

    diff_rew_nr = na * (results['Neutral_rew'] - results['RiskAware_rew'])
    diff_rew_nu = na * (results['Neutral_rew'] - results['RewUtility_rew'])

    print(f"- Duration of this round = {time.time() - start_time}")

    return key_value, results["Neutral_obj"], results["RewUtility_obj"], results["RiskAware_obj"], improve_obj_rn, improve_obj_ru, improve_obj_un, diff_obj_rn, diff_obj_ru, diff_obj_un, results["Neutral_rew"], results["RewUtility_rew"], results["RiskAware_rew"], improve_rew_nr, improve_rew_nu, diff_rew_nr, diff_rew_nu


def run_learning_combination(params):
    nt, ns, na, tt, ut, th, nc, l_episodes, n_batches, n_iterations, save_data, PATH = params

    if tt == 'structured':
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 'clinical':
        pr_ss_0 = numpy.round(numpy.linspace(0.657, 0.762, na), 3)
        numpy.random.shuffle(pr_ss_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        numpy.random.shuffle(pr_sp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        numpy.random.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, na), 3)
        numpy.random.shuffle(pr_ss_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        numpy.random.shuffle(pr_sp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        numpy.random.shuffle(pr_pp_1)
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
        numpy.random.shuffle(pr_ss_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        numpy.random.shuffle(pr_sp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        numpy.random.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, na), 3)
        numpy.random.shuffle(pr_ss_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        numpy.random.shuffle(pr_sp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        numpy.random.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1])
        ns=3

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
        numpy.random.shuffle(pr_ss_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        numpy.random.shuffle(pr_sp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        numpy.random.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, na), 3)
        numpy.random.shuffle(pr_ss_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        numpy.random.shuffle(pr_sp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        numpy.random.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1])
        ns=3
    elif tt == 'clinical-v2':
        pr_ss_0 = np.round(np.linspace(0.596, 0.690, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sr_0 = np.round(np.linspace(0.045, 0.061, na), 3)
        np.random.shuffle(pr_sr_0)
        pr_sp_0 = np.round(np.linspace(0.201, 0.287, na), 3)
        np.random.shuffle(pr_sp_0)
        pr_rr_0 = np.round(np.linspace(0.759, 0.822, na), 3)
        np.random.shuffle(pr_rr_0)
        pr_rp_0 = np.round(np.linspace(0.130, 0.169, na), 3)
        np.random.shuffle(pr_rp_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.733, 0.801, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sr_1 = np.round(np.linspace(0.047, 0.078, na), 3)
        np.random.shuffle(pr_sr_1)
        pr_sp_1 = np.round(np.linspace(0.115, 0.171, na), 3)
        np.random.shuffle(pr_sp_1)
        pr_rr_1 = np.round(np.linspace(0.758, 0.847, na), 3)
        np.random.shuffle(pr_rr_1)
        pr_rp_1 = np.round(np.linspace(0.121, 0.193, na), 3)
        np.random.shuffle(pr_rp_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sr_0, pr_sp_0, pr_rr_0, pr_rp_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_sp_1, pr_rr_1, pr_rp_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v3':
        pr_ss_0 = np.round(np.linspace(0.668, 0.738, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sr_0 = np.round(np.linspace(0.045, 0.061, na), 3)
        np.random.shuffle(pr_sr_0)
        pr_rr_0 = np.round(np.linspace(0.831, 0.870, na), 3)
        np.random.shuffle(pr_rr_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.782, 0.833, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sr_1 = np.round(np.linspace(0.047, 0.078, na), 3)
        np.random.shuffle(pr_sr_1)
        pr_rr_1 = np.round(np.linspace(0.807, 0.879, na), 3)
        np.random.shuffle(pr_rr_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = np.array([pr_ss_0, pr_sr_0, pr_rr_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_rr_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v4':
        pr_ss_0 = np.round(np.linspace(0.713, 0.799, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.829, 0.885, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = np.array([pr_ss_0, pr_pp_0, pr_ss_1, pr_pp_1])
        ns = 3

    key_value = f'df{df}_nt{nt}_ns{ns}_ng{ng}_nd{nd}_na{na}_tt{tt}_ut{ut}_th{th}_nc{nc}'
    rew_vals = rewards_inf(df, nt, na, ns)
    markov_matrix = get_transitions(na, ns, prob_remain, tt)
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)
    w_range = 2*ng
    w_trials = ng*ns*na

    prob_err_lr, indx_err_lr, _, obj_lr, _, obj_n = multiprocess_inf_learn_LRAPTSDE(
        n_iterations, df, nt, ns, ng, nd, na, nc, th, rew_vals, tt, markov_matrix, initial_states, ut[0], ut[1], 
        save_data, f'{PATH}inf_riskaware_{key_value}.joblib', w_range, w_trials
    )
    process_and_plot(prob_err_lr, indx_err_lr, obj_n, obj_lr, 'lr', PATH, key_value)


def run_avg_learning_combination(params):
    nt, ns, na, tt, nc, n_iterations, save_data, PATH = params

    if tt == 'structured':
        prob_remain = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 'clinical':
        pr_ss_0 = numpy.round(numpy.linspace(0.657, 0.762, na), 3)
        numpy.random.shuffle(pr_ss_0)
        pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, na), 3)
        numpy.random.shuffle(pr_sp_0)
        pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, na), 3)
        numpy.random.shuffle(pr_pp_0)
        pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, na), 3)
        numpy.random.shuffle(pr_ss_1)
        pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, na), 3)
        numpy.random.shuffle(pr_sp_1)
        pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, na), 3)
        numpy.random.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1])
        ns=3
    elif tt == 'clinical-v2':
        pr_ss_0 = np.round(np.linspace(0.596, 0.690, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sr_0 = np.round(np.linspace(0.045, 0.061, na), 3)
        np.random.shuffle(pr_sr_0)
        pr_sp_0 = np.round(np.linspace(0.201, 0.287, na), 3)
        np.random.shuffle(pr_sp_0)
        pr_rr_0 = np.round(np.linspace(0.759, 0.822, na), 3)
        np.random.shuffle(pr_rr_0)
        pr_rp_0 = np.round(np.linspace(0.130, 0.169, na), 3)
        np.random.shuffle(pr_rp_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.733, 0.801, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sr_1 = np.round(np.linspace(0.047, 0.078, na), 3)
        np.random.shuffle(pr_sr_1)
        pr_sp_1 = np.round(np.linspace(0.115, 0.171, na), 3)
        np.random.shuffle(pr_sp_1)
        pr_rr_1 = np.round(np.linspace(0.758, 0.847, na), 3)
        np.random.shuffle(pr_rr_1)
        pr_rp_1 = np.round(np.linspace(0.121, 0.193, na), 3)
        np.random.shuffle(pr_rp_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = numpy.array([pr_ss_0, pr_sr_0, pr_sp_0, pr_rr_0, pr_rp_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_sp_1, pr_rr_1, pr_rp_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v3':
        pr_ss_0 = np.round(np.linspace(0.668, 0.738, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sr_0 = np.round(np.linspace(0.045, 0.061, na), 3)
        np.random.shuffle(pr_sr_0)
        pr_rr_0 = np.round(np.linspace(0.831, 0.870, na), 3)
        np.random.shuffle(pr_rr_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.782, 0.833, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sr_1 = np.round(np.linspace(0.047, 0.078, na), 3)
        np.random.shuffle(pr_sr_1)
        pr_rr_1 = np.round(np.linspace(0.807, 0.879, na), 3)
        np.random.shuffle(pr_rr_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = np.array([pr_ss_0, pr_sr_0, pr_rr_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_rr_1, pr_pp_1])
        ns = 4
    elif tt == 'clinical-v4':
        pr_ss_0 = np.round(np.linspace(0.713, 0.799, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.829, 0.885, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = np.array([pr_ss_0, pr_pp_0, pr_ss_1, pr_pp_1])
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

 