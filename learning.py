from scipy.stats import dirichlet
import joblib
from whittle_v2 import *
from processes import *
from multiprocessing import Pool, cpu_count
import numpy as np
import joblib
import time
from Markov import *


def estimate_structured_transition_probabilities(
    arm_counts: np.ndarray
) -> np.ndarray:
    """
    Bayesian estimation of prob_remain for a single arm using conjugate Beta prior.
    
    Args:
        arm_counts: Counts for single arm, shape (num_states, num_states, 2, num_arms)
        alpha: Beta prior for staying probability
        beta: Beta prior for leaving probability
    
    Returns:
        Estimated prob_remain parameter
    """
    num_states = arm_counts.shape[0]
    num_arms = arm_counts.shape[-1]

    alpha = np.zeros(num_arms)
    beta = np.zeros(num_arms)

    prob_remain = np.zeros(num_arms)
    for a in range(num_arms):

        # print(f"Arm {a}:")
        # print(f"Counts (0): {arm_counts[:, :, 0, a]}")
        # print(f"Counts (1): {arm_counts[:, :, 0, a]}")

        # Count transitions that follow the "stay" pattern vs "leave" pattern
        stay_count = 0
        leave_count = 0
        
        # For action 0 transitions (more complex pattern)
        # State 0 always stays at 0
        # For other states, we need to infer from the pattern
        for s in range(1, num_states):
            # Diagonal elements have probability (num_states - s) * p
            stay_count += arm_counts[s, s, 0, a] / (num_states - s)
            # Transitions to state 0
            leave_count += arm_counts[s, 0, 0, a]
        
        # For action 1 transitions
        for s in range(num_states - 1):
            # Expected stay probability: (num_states - s - 1) * p
            # Count actual stays
            stay_count += arm_counts[s, s, 1, a]
            # Count transitions to last state (leaves)
            leave_count += arm_counts[s, num_states - 1, 1, a]
        
        # Posterior Beta distribution parameters
        alpha[a] = stay_count
        beta[a] = leave_count

        # Return a sample from posterior
        # Posterior mean: alpha[a] / (alpha[a] + beta[a])
        prob_remain[a] = np.random.beta(alpha[a], beta[a]) 
    
    # print(f"ALPHA = {alpha}")
    # print(f"BETA = {beta}")
    # print(prob_remain)
    return get_transitions(num_arms, num_states, prob_remain, 'structured')


def process_learn_LRNPTS_iteration(i, l_episodes, n_batches, n_steps, n_states, n_arms, n_choices, threshold, 
                                   true_rew, trans_type, true_dyn, initial_states, u_type, u_order, plan_wip, w_range, w_trials):

    # Initialization
    print(f"Iteration {i} starts ...")
    start_time = time.time()
    results = {
        "plan_rewards": np.zeros((l_episodes, n_arms)),
        "plan_objectives": np.zeros((l_episodes, n_arms)),
        "learn_rewards": np.zeros((l_episodes, n_arms)),
        "learn_objectives": np.zeros((l_episodes, n_arms)),
        "learn_indexerrors": np.zeros((l_episodes, n_arms)),
        "learn_transitionerrors": np.ones((l_episodes, n_arms)),
    }

    # Set up learning dynamics
    counts = np.ones((n_states, n_states, 2, n_arms))
    if trans_type == 'notfornow':
        est_transitions = estimate_structured_transition_probabilities(counts)
    else:
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for x in range(n_states):
                for act in range(2):
                    est_transitions[x, :, act, a] = dirichlet.rvs(np.ones(n_states))[0]
    lern_wip = Whittle(n_states, n_arms, true_rew, est_transitions, n_steps)
    lern_wip.get_indices(w_range, w_trials)

    for l in range(l_episodes):
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            process_neutral_whittle_learning(plan_wip, lern_wip, n_batches, n_steps, n_states, n_arms, n_choices, threshold, true_rew, trans_type, true_dyn, initial_states, u_type, u_order)
        counts += cnts

        # Update transitions
        if trans_type == 'notfornow':
            est_transitions = estimate_structured_transition_probabilities(counts)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for x in range(n_states):
                    for act in range(2):
                        est_transitions[x, :, act, a] = dirichlet.rvs(counts[x, :, act, a])[0]
        lern_wip = Whittle(n_states, n_arms, true_rew, est_transitions, n_steps)
        lern_wip.get_indices(w_range, w_trials)

        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(lern_wip.whittle_indices[a] - plan_wip.whittle_indices[a]))
            results["plan_rewards"][l, a] = np.mean(plan_totalrewards[a, :])
            results["plan_objectives"][l, a] = np.mean(plan_objectives[a, :])
            results["learn_rewards"][l, a] = np.mean(learn_totalrewards[a, :])
            results["learn_objectives"][l, a] = np.mean(learn_objectives[a, :])

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results

def multiprocess_learn_LRNPTS(
        n_iterations, l_episodes, n_batches, n_steps, n_states, n_arms, n_choices, threshold, true_rew, 
        trans_type, true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    plan_wip = Whittle(n_states, n_arms, true_rew, true_dyn, n_steps)
    plan_wip.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_batches, n_steps, n_states, n_arms, n_choices, threshold, true_rew, trans_type, true_dyn, initial_states, u_type, u_order, plan_wip, w_range, w_trials) 
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.starmap(process_learn_LRNPTS_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])

    if save_data:
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives], filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives



def process_learn_LRAPTS_iteration(i, l_episodes, n_batches, n_steps, n_states, n_arms, n_choices, threshold, true_rew, trans_type, true_dyn, initial_states, u_type, u_order, 
                                   plan_wip, w_range, w_trials):

    # Initialization
    print(f"Iteration {i} starts ...")
    start_time = time.time()
    results = {
        "plan_rewards": np.zeros((l_episodes, n_arms)),
        "plan_objectives": np.zeros((l_episodes, n_arms)),
        "learn_rewards": np.zeros((l_episodes, n_arms)),
        "learn_objectives": np.zeros((l_episodes, n_arms)),
        "learn_indexerrors": np.zeros((l_episodes, n_arms)),
        "learn_transitionerrors": np.ones((l_episodes, n_arms)),
    }

    # Set up learning dynamics
    counts = np.ones((n_states, n_states, 2, n_arms))
    if trans_type == 'notfornow':
        est_transitions = estimate_structured_transition_probabilities(counts)
    else:
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for x in range(n_states):
                for act in range(2):
                    est_transitions[x, :, act, a] = dirichlet.rvs(np.ones(n_states))[0]
    lern_wip = RiskAwareWhittle(n_states, n_arms, true_rew, est_transitions, n_steps, u_type, u_order, threshold)
    lern_wip.get_indices(w_range, w_trials)

    for l in range(l_episodes):
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            process_riskaware_whittle_learning(plan_wip, lern_wip, n_batches, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order)
        counts += cnts

        # Update transitions
        if trans_type == 'notfornow':
            est_transitions = estimate_structured_transition_probabilities(counts)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for act in range(2):
                    # print('='*10)
                    # print(f"Arm = {a}")
                    # print(f"action : {act}")
                    # print(counts[:, :, act, a])
                    for x in range(n_states):
                        est_transitions[x, :, act, a] = dirichlet.rvs(counts[x, :, act, a])[0]
        lern_wip = RiskAwareWhittle(n_states, n_arms, true_rew, est_transitions, n_steps, u_type, u_order, threshold)
        lern_wip.get_indices(w_range, w_trials)

        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(lern_wip.whittle_indices[a] - plan_wip.whittle_indices[a]))
            results["plan_rewards"][l, a] = np.mean(plan_totalrewards[a, :])
            results["plan_objectives"][l, a] = np.mean(plan_objectives[a, :])
            results["learn_rewards"][l, a] = np.mean(learn_totalrewards[a, :])
            results["learn_objectives"][l, a] = np.mean(learn_objectives[a, :])

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results

def multiprocess_learn_LRAPTS(
        n_iterations, l_episodes, n_batches, n_steps, n_states, n_arms, n_choices, threshold, true_rew, 
        trans_type, true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    plan_wip = RiskAwareWhittle(n_states, n_arms, true_rew, true_dyn, n_steps, u_type, u_order, threshold)
    plan_wip.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_batches, n_steps, n_states, n_arms, n_choices, threshold, true_rew, trans_type, true_dyn, initial_states, u_type, u_order, plan_wip, w_range, w_trials) 
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.starmap(process_learn_LRAPTS_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])

    if save_data:
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives], filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives



def process_ns_learn_LRAPTS_iteration(i, l_episodes, n_batches, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
                                      true_rew, trans_type, true_dyn, initial_states, u_type, u_order, plan_wip, w_range, w_trials):

    # Initialization
    print(f"Iteration {i} starts ...")
    start_time = time.time()
    results = {
        "plan_rewards": np.zeros((l_episodes, n_arms)),
        "plan_objectives": np.zeros((l_episodes, n_arms)),
        "learn_rewards": np.zeros((l_episodes, n_arms)),
        "learn_objectives": np.zeros((l_episodes, n_arms)),
        "learn_indexerrors": np.zeros((l_episodes, n_arms)),
        "learn_transitionerrors": np.ones((l_episodes, n_arms)),
    }

    # Set up learning dynamics
    counts = np.ones((n_states, n_states, 2, n_arms))
    if trans_type == 'notfornow':
        est_transitions = estimate_structured_transition_probabilities(counts)
    else:
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for x in range(n_states):
                for act in range(2):
                    est_transitions[x, :, act, a] = dirichlet.rvs(np.ones(n_states))[0]
    lern_wip = RiskAwareWhittleNS([n_states, n_augmnts], n_arms, true_rew, est_transitions, n_steps, u_type, u_order, threshold)
    lern_wip.get_indices(w_range, w_trials)
    counts = np.ones((n_states, n_states, 2, n_arms))

    for l in range(l_episodes):
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            process_ns_riskaware_whittle_learning(plan_wip, lern_wip, n_batches, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order)
        counts += cnts

        # Update transitions
        if trans_type == 'notfornow':
            est_transitions = estimate_structured_transition_probabilities(counts)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for act in range(2):
                    # print('='*10)
                    # print(f"Arm = {a}")
                    # print(f"action : {act}")
                    # print(counts[:, :, act, a])
                    for x in range(n_states):
                        est_transitions[x, :, act, a] = dirichlet.rvs(counts[x, :, act, a])[0]
        lern_wip = RiskAwareWhittleNS([n_states, n_augmnts], n_arms, true_rew, est_transitions, n_steps, u_type, u_order, threshold)
        lern_wip.get_indices(w_range, w_trials)

        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(lern_wip.whittle_indices[a] - plan_wip.whittle_indices[a]))
            results["plan_rewards"][l, a] = np.mean(plan_totalrewards[a, :])
            results["plan_objectives"][l, a] = np.mean(plan_objectives[a, :])
            results["learn_rewards"][l, a] = np.mean(learn_totalrewards[a, :])
            results["learn_objectives"][l, a] = np.mean(learn_objectives[a, :])
            print(f"Ite {i} - Arm {a} - AVG {np.mean(plan_totalrewards[a, :])} - STD {np.std(plan_totalrewards[a, :])}")

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results

def multiprocess_ns_learn_LRAPTS(
        n_iterations, l_episodes, n_batches, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
        true_rew, trans_type, true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    plan_wip = RiskAwareWhittleNS([n_states, n_augmnts], n_arms, true_rew, true_dyn, n_steps, u_type, u_order, threshold)
    plan_wip.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_batches, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, true_rew, trans_type, true_dyn, initial_states, 
         u_type, u_order, plan_wip, w_range, w_trials) 
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.starmap(process_ns_learn_LRAPTS_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_learn_objectives = np.stack([res["learn_objectives"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])
    all_plan_objectives = np.stack([res["plan_objectives"] for res in results])
    # print("ALL LEARNING") 
    # [print(all_learn_objective) for all_learn_objective in np.cumsum(np.sum(all_learn_objectives, axis=2), axis=1)]
    # print(f"AVG - {np.mean(np.cumsum(np.sum(all_learn_objectives, axis=2), axis=1), axis=0)}")
    # print(f"STD - {np.std(np.cumsum(np.sum(all_learn_objectives, axis=2), axis=1), axis=0)}") 
    # print("ALL PLANNING") 
    # [print(all_plan_objective) for all_plan_objective in np.cumsum(np.sum(all_plan_objectives, axis=2), axis=1)]
    # print(f"AVG - {np.mean(np.cumsum(np.sum(all_plan_objectives, axis=2), axis=1), axis=0)}")
    # print(f"STD - {np.std(np.cumsum(np.sum(all_plan_objectives, axis=2), axis=1), axis=0)}") 
    # print("ALL REGRET") 
    # [print(all_reg_objective) for all_reg_objective in np.cumsum(np.sum(all_plan_objectives - all_learn_objectives, axis=2), axis=1)]
    # print(f"AVG - {np.mean(np.cumsum(np.sum(all_plan_objectives - all_learn_objectives, axis=2), axis=1), axis=0)}")
    # print(f"STD - {np.std(np.cumsum(np.sum(all_plan_objectives - all_learn_objectives, axis=2), axis=1), axis=0)}") 

    if save_data:
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives], filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def check_episode_condition(counts, counts_at_episode_start):
    """
    Checks the count-doubling condition from RB-TSDE Algorithm 1, Line 4.
    Condition: 2 * N_tk(s, a) < N_t(s, a) for any arm i, state s, action a.
    Where N(s, a) = sum over next states s' of counts[s, s', a]
    """
    current_n_sa = np.sum(counts, axis=1) # Sum over next_state index (axis 1) -> shape (n_states, 2, n_arms)
    start_n_sa = np.sum(counts_at_episode_start, axis=1) # Sum over next_state index (axis 1)

    # Avoid division by zero or checking for counts that started at zero
    # We only check where start_n_sa is positive.
    # The condition is 2 * start < current, or current / start > 2
    relevant_indices = start_n_sa > 0
    if not np.any(relevant_indices): # If no counts at the start, condition is trivially false
        return False

    ratio = np.full_like(current_n_sa, 0.0) # Initialize ratio array
    # Calculate ratio only where start_n_sa is positive
    ratio[relevant_indices] = current_n_sa[relevant_indices] / start_n_sa[relevant_indices]

    # Check if any ratio exceeds 2
    return np.any(ratio > 2)

def process_inf_learn_LRAPNTSDE_iteration(i, discount, n_steps, n_states, n_arms, n_choices, threshold, true_rew, trans_type, true_dyn, 
                                          initial_states, u_type, u_order, w_range, w_trials):

    # Initialization
    print(f"Iteration {i} (TSDE) starts ...")
    start_time = time.time()
    results = {
        "learn_rewards": np.zeros((n_steps, n_arms)),
        "learn_objectives": np.zeros((n_steps, n_arms)),
        "learn_transitionerrors": np.ones((n_steps, n_arms)),
    }

    # Initialize counts (posterior beliefs) - start with prior (e.g., 1)
    counts = np.ones((n_states, n_states, 2, n_arms))

    # --- TSDE Specific Initialization ---
    k = 0  # Episode counter
    t_k = 0  # Start time of current episode k
    T_k_minus_1 = 0  # Initialize length of previous episode (T_{k-1})
    counts_at_episode_start = np.copy(counts) # Store counts at t=0 (start of episode 0)

    # Initial policy computation (Start of Episode 0)
    # Sample initial transitions from the prior
    if trans_type == 'notfornow':
        est_transitions = estimate_structured_transition_probabilities(counts)
    else:
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for x in range(n_states):
                for act in range(2):
                    est_transitions[x, :, act, a] = dirichlet.rvs(np.ones(n_states))[0]
    # Create lern_wip object based on the *sampled* transitions for this episode
    lern_wip = WhittleInf(n_states, n_arms, true_rew, est_transitions, n_steps, discount)
    lern_wip.get_indices(w_range, w_trials)
    # ------------------------------------

    sample_paths = 1
    learn_totalrewards = np.zeros((n_arms, sample_paths))
    learn_utility = np.zeros((n_arms, sample_paths))
    learn_states = (n_states - 1) * np.ones((n_arms, sample_paths), dtype=np.int32)

    for t in range(n_steps):

        # --- TSDE Episode Check ---
        # Check if a new episode should start (using counts from the *end* of step t-1)
        # The check uses N_t(s,a) which corresponds to counts *before* processing step t

        start_new_episode = False
        # Condition 1: Count doubling
        count_condition_met = check_episode_condition(counts, counts_at_episode_start)
        # Condition 2: Current episode length exceeds previous episode length
        current_episode_length = t - t_k
        # Note: T_k_minus_1 stores the length of the episode that *ended* before the current one started
        length_condition_met = (current_episode_length > T_k_minus_1)

        # Check if *either* condition is met
        if count_condition_met:
            start_new_episode = True
            reason = "Count Doubling"
        elif length_condition_met:
            start_new_episode = True
            reason = f"Length ({current_episode_length} > {T_k_minus_1})"

        # if start_new_episode and t > t_k: # <<< Keep this outer check (redundant now but safe)
        if start_new_episode: # <<< Simplified check, as t > t_k is handled above
            T_k = t - t_k # Length of the completed episode k
            T_k_minus_1 = T_k # Update T_{k-1} for the *next* episode's check
            k += 1
            t_k = t # New episode starts *at* the beginning of step t
            counts_at_episode_start = np.copy(counts) # Store current counts
            # print(f"  t={t}: Starting Episode k={k} (Reason: {reason})") # <<< Updated Print Statement

            # Sample new transitions from the updated posterior (counts)
            if trans_type == 'notfornow':
                est_transitions = estimate_structured_transition_probabilities(counts)
            else:
                est_transitions = np.zeros((n_states, n_states, 2, n_arms))
                for a in range(n_arms):
                    for act in range(2):
                        # print('='*10)
                        # print(f"Arm = {a}")
                        # print(f"action : {act}")
                        # print(counts[:, :, act, a])
                        for x in range(n_states):
                            est_transitions[x, :, act, a] = dirichlet.rvs(counts[x, :, act, a])[0]

            # Update the lern_wip object 
            lern_wip = WhittleInf(n_states, n_arms, true_rew, est_transitions, n_steps, discount)
            lern_wip.get_indices(w_range, w_trials)

        # --- End TSDE Episode Check ---

        # Use the policy computed at the start of the *current* episode k
        discount_val = discount ** t
        for s in range(sample_paths):
            learn_actions = lern_wip.take_action(n_choices, {"x": learn_states[:, s]})
            _learn_states = np.copy(learn_states[:, s])
            for a in range(n_arms):
                learn_totalrewards[a, s] += discount_val * true_rew[learn_states[a, s], a]
                learn_utility[a, s] = compute_utility(learn_totalrewards[a, s], threshold, u_type, u_order)
                learn_states[a, s] = np.random.choice(n_states, p=true_dyn[learn_states[a, s], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a, s], learn_actions[a], a] += 1
        for a in range(n_arms):
            results["learn_transitionerrors"][t, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_rewards"][t, a] = np.mean(learn_totalrewards[a, :])
            results["learn_objectives"][t, a] = np.mean(learn_utility[a, :])
        # print(f"t={t}, a={a}, states={states[a]}, learn_states={learn_states[a]}")
        # print(f"t={t}, a={a}, plan_rew={true_rew[states[a], a]}, learn_rew={true_rew[learn_states[a], a]}, discount_val={discount_val}")
        # print(f"t={t}, a={a}, plan_rewards={results['plan_rewards'][t, a]}, learn_rewards={results['learn_rewards'][t, a]}")
        # print(f"t={t}, a={a}, plan_objective={results["plan_objectives"][t, a]}, learn_objective={results["learn_objectives"][t, a]}")

    print(f"Iteration {i} (TSDE) end with duration: {time.time() - start_time}")
    return results

def process_inf_learn_LRAPTSDE_iteration(i, discount, n_steps, n_states, n_augmnts, n_discounts, n_arms, n_choices, threshold,
                                         true_rew, trans_type, true_dyn, initial_states, u_type, u_order, plan_wip, plan_rawip, 
                                         w_range, w_trials):

    # Initialization
    print(f"Iteration {i} (TSDE) starts ...")
    start_time = time.time()
    results = {
        "learn_rewards": np.zeros((n_steps, n_arms)),
        "learn_objectives": np.zeros((n_steps, n_arms)),
        "learn_indexerrors": np.zeros((n_steps, n_arms)),
        "learn_transitionerrors": np.ones((n_steps, n_arms)),
    }

    # Initialize counts (posterior beliefs) - start with prior (e.g., 1)
    counts = np.ones((n_states, n_states, 2, n_arms))

    # --- TSDE Specific Initialization ---
    k = 0  # Episode counter
    t_k = 0  # Start time of current episode k
    T_k_minus_1 = 0  # Initialize length of previous episode (T_{k-1})
    counts_at_episode_start = np.copy(counts) # Store counts at t=0 (start of episode 0)

    # Initial policy computation (Start of Episode 0)
    # Sample initial transitions from the prior
    if trans_type == 'notfornow':
        est_transitions = estimate_structured_transition_probabilities(counts)
    else:
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for x in range(n_states):
                for act in range(2):
                    est_transitions[x, :, act, a] = dirichlet.rvs(np.ones(n_states))[0]
    # Create lern_wip object based on the *sampled* transitions for this episode
    lern_rawip = RiskAwareWhittleInf([n_states, n_augmnts, n_discounts], n_arms, true_rew, est_transitions, discount, u_type, u_order, threshold)
    lern_rawip.get_indices(w_range, w_trials)
    # ------------------------------------

    sample_paths = 1
    learn_totalrewards = np.zeros((n_arms, sample_paths))
    learn_utility = np.zeros((n_arms, sample_paths))
    learn_lifted = np.zeros((n_arms, sample_paths), dtype=np.int32)
    learn_states = (n_states - 1) * np.ones((n_arms, sample_paths), dtype=np.int32)

    for t in range(n_steps):

        # --- TSDE Episode Check ---
        # Check if a new episode should start (using counts from the *end* of step t-1)
        # The check uses N_t(s,a) which corresponds to counts *before* processing step t

        start_new_episode = False
        # Condition 1: Count doubling
        count_condition_met = check_episode_condition(counts, counts_at_episode_start)
        # Condition 2: Current episode length exceeds previous episode length
        current_episode_length = t - t_k
        # Note: T_k_minus_1 stores the length of the episode that *ended* before the current one started
        length_condition_met = (current_episode_length > T_k_minus_1)

        # Check if *either* condition is met
        if count_condition_met:
            start_new_episode = True
            reason = "Count Doubling"
        elif length_condition_met:
            start_new_episode = True
            reason = f"Length ({current_episode_length} > {T_k_minus_1})"

        # if start_new_episode and t > t_k: # <<< Keep this outer check (redundant now but safe)
        if start_new_episode: # <<< Simplified check, as t > t_k is handled above
            T_k = t - t_k # Length of the completed episode k
            T_k_minus_1 = T_k # Update T_{k-1} for the *next* episode's check
            k += 1
            t_k = t # New episode starts *at* the beginning of step t
            counts_at_episode_start = np.copy(counts) # Store current counts
            # print(f"  t={t}: Starting Episode k={k} (Reason: {reason})") # <<< Updated Print Statement

            # Sample new transitions from the updated posterior (counts)
            if trans_type == 'notfornow':
                est_transitions = estimate_structured_transition_probabilities(counts)
            else:
                est_transitions = np.zeros((n_states, n_states, 2, n_arms))
                for a in range(n_arms):
                    for act in range(2):
                        # print('='*10)
                        # print(f"Arm = {a}")
                        # print(f"action : {act}")
                        # print(counts[:, :, act, a])
                        for x in range(n_states):
                            est_transitions[x, :, act, a] = dirichlet.rvs(counts[x, :, act, a])[0]

            # Update the lern_wip object with the *newly sampled* transitions for this episode
            if t < n_discounts:
                lern_rawip = RiskAwareWhittleInf([n_states, n_augmnts, n_discounts], n_arms, true_rew, est_transitions, discount, u_type, u_order, threshold)
                lern_rawip.get_indices(w_range, w_trials)
                lern_wip = WhittleInf(n_states, n_arms, true_rew, est_transitions, n_steps, discount)
                lern_wip.get_indices(w_range, w_trials)
                # for a in range(n_arms):
                #     for x in range(n_states):
                #         lern_rawip.whittle_indices[a][:, x, :] = lern_rawip.whittle_indices[a][:, x, :] + np.sqrt( 2 * np.log( sum(counts[x, :, 0, a] + counts[x, :, 1, a]) ) / sum(counts[x, :, 1, a]) )
                #         lern_rawip.whittle_indices[a][:, x, :] = np.maximum(0, lern_rawip.whittle_indices[a][:, x, :] - np.sqrt( 2 * np.log( sum(counts[x, :, 0, a] + counts[x, :, 1, a]) ) / sum(counts[x, :, 0, a]) ))
                #         lern_wip.whittle_indices[a][x] = lern_wip.whittle_indices[a][x] + np.sqrt( 2 * np.log( sum(counts[x, :, 0, a] + counts[x, :, 1, a]) ) / sum(counts[x, :, 1, a]) )
                #         lern_wip.whittle_indices[a][x] = np.maximum(0, lern_wip.whittle_indices[a][x] - np.sqrt( 2 * np.log( sum(counts[x, :, 0, a] + counts[x, :, 1, a]) ) / sum(counts[x, :, 0, a]) ))

            else:
                lern_wip = WhittleInf(n_states, n_arms, true_rew, est_transitions, n_steps, discount)
                lern_wip.get_indices(w_range, w_trials)

                # for a in range(n_arms):
                #     for x in range(n_states):
                #         lern_wip.whittle_indices[a][x] = lern_wip.whittle_indices[a][x] + np.sqrt( 2 * np.log( sum(counts[x, :, 0, a] + counts[x, :, 1, a]) ) / sum(counts[x, :, 1, a]) )
                #         lern_wip.whittle_indices[a][x] = np.maximum(0, lern_wip.whittle_indices[a][x] - np.sqrt( 2 * np.log( sum(counts[x, :, 0, a] + counts[x, :, 1, a]) ) / sum(counts[x, :, 0, a]) ))

        # --- End TSDE Episode Check ---

        # Use the policy computed at the start of the *current* episode k
        discount_val = discount ** t
        for s in range(sample_paths):
            if t < n_discounts:
                learn_actions = lern_rawip.take_action(n_choices, {"l": learn_lifted[:, s], "x": learn_states[:, s], "t": t})
            else:
                learn_actions = lern_wip.take_action(n_choices, {"x": learn_states[:, s]})
            _learn_states = np.copy(learn_states[:, s])
            for a in range(n_arms):
                learn_totalrewards[a, s] += discount_val * true_rew[learn_states[a, s], a]
                learn_utility[a, s] = compute_utility(learn_totalrewards[a, s], threshold, u_type, u_order)
                if t < n_discounts:
                    learn_lifted[a, s] = lern_rawip.get_reward_partition(learn_totalrewards[a, s])
                learn_states[a, s] = np.random.choice(n_states, p=true_dyn[learn_states[a, s], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a, s], learn_actions[a], a] += 1
        for a in range(n_arms):
            results["learn_transitionerrors"][t, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_rewards"][t, a] = np.mean(learn_totalrewards[a, :])
            results["learn_objectives"][t, a] = np.mean(learn_utility[a, :])
            if t < n_discounts:
                results["learn_indexerrors"][t, a] = np.max(np.abs(lern_rawip.whittle_indices[a] - plan_rawip.whittle_indices[a]))
            else:
                results["learn_indexerrors"][t, a] = np.max(np.abs(lern_wip.whittle_indices[a] - plan_wip.whittle_indices[a]))
        # print(f"t={t}, a={a}, states={states[a]}, learn_states={learn_states[a]}")
        # print(f"t={t}, a={a}, plan_rew={true_rew[states[a], a]}, learn_rew={true_rew[learn_states[a], a]}, discount_val={discount_val}")
        # print(f"t={t}, a={a}, plan_rewards={results['plan_rewards'][t, a]}, learn_rewards={results['learn_rewards'][t, a]}")
        # print(f"t={t}, a={a}, plan_objective={results["plan_objectives"][t, a]}, learn_objective={results["learn_objectives"][t, a]}")

    print(f"Iteration {i} (TSDE) end with duration: {time.time() - start_time}")
    return results

def multiprocess_inf_learn_LRAPTSDE(
        n_iterations, discount, n_steps, n_states, n_augmnts, n_discounts, n_arms, n_choices, threshold, 
        true_rew, trans_type, true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    plan_wip = WhittleInf(n_states, n_arms, true_rew, true_dyn, n_steps, discount)
    plan_wip.get_indices(w_range, w_trials)
    
    plan_rawip = RiskAwareWhittleInf([n_states, n_augmnts, n_discounts], n_arms, true_rew, true_dyn, discount, u_type, u_order, threshold)
    plan_rawip.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, discount, n_steps, n_states, n_augmnts, n_discounts, n_arms, n_choices, threshold, true_rew, trans_type, true_dyn, initial_states, u_type, 
         u_order, plan_wip, plan_rawip, w_range, w_trials) 
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        riskaware_res = pool.starmap(process_inf_learn_LRAPTSDE_iteration, args)

    # Aggregate results
    riskaware_results = {}
    riskaware_results["transitionerrors"] = np.stack([res["learn_transitionerrors"] for res in riskaware_res])
    riskaware_results["indexerrors"] = np.stack([res["learn_indexerrors"] for res in riskaware_res])
    riskaware_results["rewards"] = np.stack([res["learn_rewards"] for res in riskaware_res])
    riskaware_results["objectives"] = np.stack([res["learn_objectives"] for res in riskaware_res])

    # Define arguments for each iteration
    args = [
        (i, discount, n_steps, n_states, n_arms, n_choices, threshold, true_rew, trans_type, true_dyn, initial_states, u_type, 
         u_order, w_range, w_trials) 
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        neutral_res = pool.starmap(process_inf_learn_LRAPNTSDE_iteration, args)

    # Aggregate results
    neutral_results = {}
    neutral_results["transitionerrors"] = np.stack([res["learn_transitionerrors"] for res in neutral_res])
    neutral_results["rewards"] = np.stack([res["learn_rewards"] for res in neutral_res])
    neutral_results["objectives"] = np.stack([res["learn_objectives"] for res in neutral_res])

    # Define all processes to evaluate
    processes = [
        # ("RND", lambda *args: process_inf_random_policy(*args)),
        # ("MYP", lambda *args: process_inf_myopic_policy(*args)),
        # ("WIP", lambda *args: process_inf_neutral_whittle(plan_wip, *args)),
        ("RAP", lambda *args: process_inf_riskaware_whittle(plan_rawip, plan_wip, n_discounts, *args))
    ]

    # Run all processes and collect results
    baseline_results = {}
    common_args = (n_iterations, discount, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order)
    
    for name, process in processes:
        rew, obj = process(*common_args)
        baseline_results[f"{name}_rew"] = rew
        baseline_results[f"{name}_obj"] = obj

    if save_data:
        joblib.dump([riskaware_results, neutral_results, baseline_results], filename)

    return riskaware_results, neutral_results, baseline_results


def process_avg_learn_TSDE_iteration(i, n_steps, n_states, n_arms, n_choices, true_rew, trans_type, true_dyn, initial_states, plan_wip, w_range, w_trials):
    # Initialization
    print(f"Iteration {i} (TSDE) starts ...")
    start_time = time.time()
    results = {
        "plan_rewards": np.zeros((n_steps, n_arms)),
        "plan_objectives": np.zeros((n_steps, n_arms)),
        "learn_rewards": np.zeros((n_steps, n_arms)),
        "learn_objectives": np.zeros((n_steps, n_arms)),
        "learn_indexerrors": np.zeros((n_steps, n_arms)),
        "learn_transitionerrors": np.ones((n_steps, n_arms)),
    }

    # Initialize counts (posterior beliefs) - start with prior (e.g., 1)
    counts = np.ones((n_states, n_states, 2, n_arms), dtype=np.int16)

    # --- TSDE Specific Initialization ---
    k = 0  # Episode counter
    t_k = 0  # Start time of current episode k
    T_k_minus_1 = 0  # Initialize length of previous episode (T_{k-1})
    counts_at_episode_start = np.copy(counts) # Store counts at t=0 (start of episode 0)
    est_transitions = np.zeros((n_states, n_states, 2, n_arms)) # To store the sampled transitions for the episode

    # Initial policy computation (Start of Episode 0)
    # print("t=0: Starting Episode k=0")
    # Sample initial transitions from the prior
    if trans_type == 'notfornow':
        est_transitions = estimate_structured_transition_probabilities(counts)
    else:
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for act in range(2):
                for x in range(n_states):
                    est_transitions[x, :, act, a] = dirichlet.rvs(counts[x, :, act, a])[0]

    discount = 0.999

    # Create lern_wip object based on the *sampled* transitions for this episode
    lern_wip = WhittleInf(n_states, n_arms, true_rew, est_transitions, n_steps, discount)
    lern_wip.get_indices(w_range, w_trials) # Compute policy (Whittle indices) for the episode
    # ------------------------------------

    sample_paths = 1
    plan_rewards = np.zeros((n_arms, sample_paths))
    learn_rewards = np.zeros((n_arms, sample_paths))
    states = (n_states - 1) * np.ones((n_arms, sample_paths), dtype=np.int32)
    learn_states = (n_states - 1) * np.ones((n_arms, sample_paths), dtype=np.int32)

    for t in range(n_steps):

        # --- TSDE Episode Check ---
        # Check if a new episode should start (using counts from the *end* of step t-1)
        # The check uses N_t(s,a) which corresponds to counts *before* processing step t

        start_new_episode = False
        # Condition 1: Count doubling
        count_condition_met = check_episode_condition(counts, counts_at_episode_start)
        # Condition 2: Current episode length exceeds previous episode length
        current_episode_length = t - t_k
        # Note: T_k_minus_1 stores the length of the episode that *ended* before the current one started
        length_condition_met = (current_episode_length > T_k_minus_1)

        # Check if *either* condition is met
        if count_condition_met:
            start_new_episode = True
            reason = "Count Doubling"
        elif length_condition_met:
            start_new_episode = True
            reason = f"Length ({current_episode_length} > {T_k_minus_1})"

        # if start_new_episode and t > t_k: # <<< Keep this outer check (redundant now but safe)
        if start_new_episode: # <<< Simplified check, as t > t_k is handled above
            T_k = t - t_k # Length of the completed episode k
            T_k_minus_1 = T_k # Update T_{k-1} for the *next* episode's check
            k += 1
            t_k = t # New episode starts *at* the beginning of step t
            counts_at_episode_start = np.copy(counts) # Store current counts
            print(f"  t={t}: Starting Episode k={k} (Reason: {reason})") # <<< Updated Print Statement

            # Sample new transitions from the updated posterior (counts)
            if trans_type == 'notfornow':
                est_transitions = estimate_structured_transition_probabilities(counts)
            else:
                est_transitions = np.zeros((n_states, n_states, 2, n_arms))
                for a in range(n_arms):
                    for act in range(2):
                        # print('='*10)
                        # print(f"Arm = {a}")
                        # print(f"action : {act}")
                        # print(counts[:, :, act, a])
                        for x in range(n_states):
                            est_transitions[x, :, act, a] = dirichlet.rvs(counts[x, :, act, a])[0]

            # Update the lern_wip object with the *newly sampled* transitions for this episode
            lern_wip = WhittleInf(n_states, n_arms, true_rew, est_transitions, n_steps, discount)
            lern_wip.get_indices(w_range, w_trials)

            for a in range(n_arms):
                for x in range(n_states):
                    lern_wip.whittle_indices[a][x] = lern_wip.whittle_indices[a][x] + np.sqrt( 2 * np.log( sum(counts[x, :, 0, a] + counts[x, :, 1, a]) ) / sum(counts[x, :, 1, a]) )
                    lern_wip.whittle_indices[a][x] = np.maximum(0, lern_wip.whittle_indices[a][x] - np.sqrt( 2 * np.log( sum(counts[x, :, 0, a] + counts[x, :, 1, a]) ) / sum(counts[x, :, 0, a]) ))

        # --- End TSDE Episode Check ---

        # Use the policy computed at the start of the *current* episode k
        discount_val = discount ** t
        for s in range(sample_paths):
            actions = plan_wip.take_action(n_choices, {"x": states[:, s]})
            learn_actions = lern_wip.take_action(n_choices, {"x": learn_states[:, s]})
            _learn_states = np.copy(learn_states[:, s])
            for a in range(n_arms):
                plan_rewards[a, s] = discount_val * true_rew[states[a, s], a]
                learn_rewards[a, s] = discount_val * true_rew[learn_states[a, s], a]
                states[a, s] = np.random.choice(n_states, p=true_dyn[states[a, s], :, actions[a], a])
                learn_states[a, s] = np.random.choice(n_states, p=true_dyn[learn_states[a, s], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a, s], learn_actions[a], a] += 1
        for a in range(n_arms):
            results["learn_transitionerrors"][t, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["plan_rewards"][t, a] = np.mean(plan_rewards[a, :])
            results["learn_rewards"][t, a] = np.mean(learn_rewards[a, :])
            results["learn_indexerrors"][t, a] = np.max(np.abs(lern_wip.whittle_indices[a] - plan_wip.whittle_indices[a]))

    print(f"Iteration {i} (TSDE) end with duration: {time.time() - start_time}")
    return results

def multiprocess_avg_learn_TSDE(
        n_iterations, n_steps, n_states, n_arms, n_choices, 
        true_rew, trans_type, true_dyn, initial_states, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    plan_wip = WhittleInf(n_states, n_arms, true_rew, true_dyn, n_steps, discount=0.99)
    plan_wip.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, n_steps, n_states, n_arms, n_choices, true_rew, trans_type, true_dyn, initial_states, plan_wip, w_range, w_trials) 
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.starmap(process_avg_learn_TSDE_iteration, args)

    # Aggregate results
    all_learn_transitionerrors = np.stack([res["learn_transitionerrors"] for res in results])
    all_learn_indexerrors = np.stack([res["learn_indexerrors"] for res in results])
    all_learn_rewards = np.stack([res["learn_rewards"] for res in results])
    all_plan_rewards = np.stack([res["plan_rewards"] for res in results])

    if save_data:
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_plan_rewards], filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_plan_rewards
