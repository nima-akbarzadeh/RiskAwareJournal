from scipy.stats import dirichlet
import joblib
from whittle import *
from processes import *
from multiprocessing import Pool, cpu_count
import numpy as np
import joblib
import time


def process_learn_LRAPTS_iteration(i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order, 
                                   PlanW, w_range, w_trials):

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
    est_transitions = np.zeros((n_states, n_states, 2, n_arms))
    for a in range(n_arms):
        for s1 in range(n_states):
            for act in range(2):
                est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
    LearnW = RiskAwareWhittle(n_states, n_arms, true_rew, est_transitions, n_steps, u_type, u_order, threshold)
    LearnW.get_indices(w_range, w_trials)
    counts = np.ones((n_states, n_states, 2, n_arms))

    for l in range(l_episodes):
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            process_riskaware_whittle_learning(PlanW, LearnW, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order)
        counts += cnts

        # Update transitions
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
        LearnW = RiskAwareWhittle(n_states, n_arms, true_rew, est_transitions, n_steps, u_type, u_order, threshold)
        LearnW.get_indices(w_range, w_trials)

        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(LearnW.whittle_indices[a] - PlanW.whittle_indices[a]))
            results["plan_rewards"][l, a] = np.mean(plan_totalrewards[a, :])
            results["plan_objectives"][l, a] = np.mean(plan_objectives[a, :])
            results["learn_rewards"][l, a] = np.mean(learn_totalrewards[a, :])
            results["learn_objectives"][l, a] = np.mean(learn_objectives[a, :])

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results


def multiprocess_learn_LRAPTS(
        n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, 
        true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    PlanW = RiskAwareWhittle(n_states, n_arms, true_rew,  true_dyn, n_steps, u_type, u_order, threshold)
    PlanW.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials) 
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


def process_learn_LRNPTS_iteration(i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, 
                                   true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials):

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
    est_transitions = np.zeros((n_states, n_states, 2, n_arms))
    for a in range(n_arms):
        for s1 in range(n_states):
            for act in range(2):
                est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
    LearnW = Whittle(n_states, n_arms, true_rew, est_transitions, n_steps)
    LearnW.get_indices(w_range, w_trials)
    counts = np.ones((n_states, n_states, 2, n_arms))

    for l in range(l_episodes):
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            process_neutral_whittle_learning(PlanW, LearnW, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order)
        counts += cnts

        # Update transitions
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
        LearnW = Whittle(n_states, n_arms, true_rew, est_transitions, n_steps)
        LearnW.get_indices(w_range, w_trials)

        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(LearnW.whittle_indices[a] - PlanW.whittle_indices[a]))
            results["plan_rewards"][l, a] = np.mean(plan_totalrewards[a, :])
            results["plan_objectives"][l, a] = np.mean(plan_objectives[a, :])
            results["learn_rewards"][l, a] = np.mean(learn_totalrewards[a, :])
            results["learn_objectives"][l, a] = np.mean(learn_objectives[a, :])

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results


def multiprocess_learn_LRNPTS(
        n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, 
        true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    PlanW = Whittle(n_states, n_arms, true_rew, true_dyn, n_steps)
    PlanW.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials) 
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


def process_ns_learn_LRAPTS_iteration(i, l_episodes, n_episodes, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
                                      true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials):

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
    est_transitions = np.zeros((n_states, n_states, 2, n_arms))
    for a in range(n_arms):
        for s1 in range(n_states):
            for act in range(2):
                est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
    LearnW = RiskAwareWhittleNS([n_states, n_augmnts], n_arms, true_rew, est_transitions, n_steps, u_type, u_order, threshold)
    LearnW.get_indices(w_range, w_trials)
    counts = np.ones((n_states, n_states, 2, n_arms))

    for l in range(l_episodes):
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            process_ns_riskaware_whittle_learning(PlanW, LearnW, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order)
        counts += cnts

        # Update transitions
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
        LearnW = RiskAwareWhittleNS([n_states, n_augmnts], n_arms, true_rew, est_transitions, n_steps, u_type, u_order, threshold)
        LearnW.get_indices(w_range, w_trials)

        for a in range(n_arms):
            results["learn_transitionerrors"][l, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_indexerrors"][l, a] = np.max(np.abs(LearnW.whittle_indices[a] - PlanW.whittle_indices[a]))
            results["plan_rewards"][l, a] = np.mean(plan_totalrewards[a, :])
            results["plan_objectives"][l, a] = np.mean(plan_objectives[a, :])
            results["learn_rewards"][l, a] = np.mean(learn_totalrewards[a, :])
            results["learn_objectives"][l, a] = np.mean(learn_objectives[a, :])

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results


def multiprocess_ns_learn_LRAPTS(
        n_iterations, l_episodes, n_episodes, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
        true_rew, true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    PlanW = RiskAwareWhittleNS([n_states, n_augmnts], n_arms, true_rew, true_dyn, n_steps, u_type, u_order, threshold)
    PlanW.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, 
         u_type, u_order, PlanW, w_range, w_trials) 
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

    if save_data:
        joblib.dump([all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives], filename)

    return all_learn_transitionerrors, all_learn_indexerrors, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives


def check_episode_condition(counts, counts_at_episode_start):
    """
    Checks the count-doubling condition from RB-TSDE Algorithm 1, Line 4.
    Condition: 2 * N_tk(s, a) < N_t(s, a) for any arm i, state s, action a.
    Where N(s, a) = sum over next states s' of counts[s, s', a]
    """
    current_N_sa = np.sum(counts, axis=1) # Sum over next_state index (axis 1) -> shape (n_states, 2, n_arms)
    start_N_sa = np.sum(counts_at_episode_start, axis=1) # Sum over next_state index (axis 1)

    # Avoid division by zero or checking for counts that started at zero
    # We only check where start_N_sa is positive.
    # The condition is 2 * start < current, or current / start > 2
    relevant_indices = start_N_sa > 0
    if not np.any(relevant_indices): # If no counts at the start, condition is trivially false
        return False

    ratio = np.full_like(current_N_sa, 0.0) # Initialize ratio array
    # Calculate ratio only where start_N_sa is positive
    ratio[relevant_indices] = current_N_sa[relevant_indices] / start_N_sa[relevant_indices]

    # Check if any ratio exceeds 2
    return np.any(ratio > 2)


def process_inf_learn_LRAPTSDE_iteration(i, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold,
                                             true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials):
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
    counts = np.ones((n_states, n_states, 2, n_arms))

    # --- TSDE Specific Initialization ---
    k = 0  # Episode counter
    t_k = 0  # Start time of current episode k
    T_k_minus_1 = 0  # Initialize length of previous episode (T_{k-1})
    counts_at_episode_start = np.copy(counts) # Store counts at t=0 (start of episode 0)
    est_transitions = np.zeros((n_states, n_states, 2, n_arms)) # To store the sampled transitions for the episode

    # Initial policy computation (Start of Episode 0)
    # print("t=0: Starting Episode k=0")
    # Sample initial transitions from the prior
    for a in range(n_arms):
        for s1 in range(n_states):
            for act in range(2):
                est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])[0] # Sample from posterior
    # Create LearnW object based on the *sampled* transitions for this episode
    LearnW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, est_transitions, discount, u_type, u_order, threshold)
    LearnW.get_indices(w_range, w_trials) # Compute policy (Whittle indices) for the episode
    # ------------------------------------

    plan_totalrewards = np.zeros(n_arms)
    learn_totalrewards = np.zeros(n_arms)
    lifted = np.zeros(n_arms, dtype=np.int32)
    states = initial_states.copy()
    learn_lifted = np.zeros(n_arms, dtype=np.int32)
    learn_states = initial_states.copy()

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
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])[0]

            # Update the LearnW object with the *newly sampled* transitions for this episode
            LearnW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, est_transitions, discount, u_type, u_order, threshold)
            LearnW.get_indices(w_range, w_trials) # Compute policy (Whittle indices) for the new episode
        # --- End TSDE Episode Check ---

        # Use the policy computed at the start of the *current* episode k
        discount_val = discount ** t
        discount_idx = PlanW.get_discnt_partition(discount_val)
        learn_discount_idx = LearnW.get_discnt_partition(discount_val)
        actions = PlanW.take_action(n_choices, lifted, discount_idx, states)
        learn_actions = LearnW.take_action(n_choices, learn_lifted, learn_discount_idx, learn_states)
        _learn_states = np.copy(learn_states)
        for a in range(n_arms):
            plan_totalrewards[a] += discount_val * true_rew[states[a], a]
            lifted[a] = PlanW.get_reward_partition(plan_totalrewards[a])
            states[a] = np.random.choice(n_states, p=true_dyn[states[a], :, actions[a], a])
            learn_totalrewards[a] += discount_val * true_rew[learn_states[a], a]
            learn_lifted[a] = LearnW.get_reward_partition(learn_totalrewards[a])
            learn_states[a] = np.random.choice(n_states, p=true_dyn[learn_states[a], :, learn_actions[a], a])
            counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
            results["learn_transitionerrors"][t, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_indexerrors"][t, a] = np.max(np.abs(LearnW.whittle_indices[a] - PlanW.whittle_indices[a]))
            results["plan_rewards"][t, a] = plan_totalrewards[a]
            results["plan_objectives"][t, a] = compute_utility(plan_totalrewards[a], threshold, u_type, u_order)
            results["learn_rewards"][t, a] = learn_totalrewards[a]
            results["learn_objectives"][t, a] = compute_utility(learn_totalrewards[a], threshold, u_type, u_order)
            print(f"t={t}, a={a}, states={states[a]}, learn_states={learn_states[a]}")
            print(f"t={t}, a={a}, plan_rew={true_rew[states[a], a]}, learn_rew={true_rew[learn_states[a], a]}, discount_val={discount_val}")
            print(f"t={t}, a={a}, plan_rewards={results['plan_rewards'][t, a]}, learn_rewards={results['learn_rewards'][t, a]}")
            print(f"t={t}, a={a}, plan_objective={results["plan_objectives"][t, a]}, learn_objective={results["learn_objectives"][t, a]}")

    print(f"Iteration {i} (TSDE) end with duration: {time.time() - start_time}")
    return results


def multiprocess_inf_learn_LRAPTSDE(
        n_iterations, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
        true_rew, true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    PlanW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, true_dyn, discount, u_type, u_order, threshold)
    PlanW.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials) 
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.starmap(process_inf_learn_LRAPTSDE_iteration, args)

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


def process_inf_learn_LRAPTS_iteration(i, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
                                       true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials): 
    # Initialization
    print(f"Iteration {i} starts ...")
    start_time = time.time()
    results = {
        "plan_rewards": np.zeros((n_steps, n_arms)),
        "plan_objectives": np.zeros((n_steps, n_arms)),
        "learn_rewards": np.zeros((n_steps, n_arms)),
        "learn_objectives": np.zeros((n_steps, n_arms)),
        "learn_indexerrors": np.zeros((n_steps, n_arms)),
        "learn_transitionerrors": np.ones((n_steps, n_arms)),
    }

    # Set up learning dynamics
    est_transitions = np.zeros((n_states, n_states, 2, n_arms))
    for a in range(n_arms):
        for s1 in range(n_states):
            for act in range(2):
                est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
    LearnW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, est_transitions, discount, n_steps, u_type, u_order, threshold)
    LearnW.get_indices(w_range, w_trials)
    counts = np.ones((n_states, n_states, 2, n_arms))

    plan_totalrewards = np.zeros(n_arms)
    learn_totalrewards = np.zeros(n_arms)
    lifted = np.zeros(n_arms, dtype=np.int32)
    states = initial_states.copy()
    learn_lifted = np.zeros(n_arms, dtype=np.int32)
    learn_states = initial_states.copy()

    for t in range(n_steps):
        
        discount_val = (1 - discount) * discount ** t
        discount_idx = PlanW.get_discnt_partition(discount_val)
        actions = PlanW.take_action(n_choices, lifted, discount_idx, states)
        learn_actions = LearnW.take_action(n_choices, learn_lifted, discount_idx, learn_states)
        _learn_states = np.copy(learn_states)
        for a in range(n_arms):
            plan_totalrewards[a] += discount_val * true_rew[states[a], a]
            lifted[a] = PlanW.get_reward_partition(plan_totalrewards[a])
            states[a] = np.random.choice(n_states, p=true_dyn[states[a], :, actions[a], a])
            learn_totalrewards[a] += discount_val * true_rew[learn_states[a], a]
            learn_lifted[a] = LearnW.get_reward_partition(learn_totalrewards[a])
            learn_states[a] = np.random.choice(n_states, p=true_dyn[learn_states[a], :, learn_actions[a], a])
            counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
        
        # Update transitions
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
        LearnW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, est_transitions, discount, n_steps, u_type, u_order, threshold)
        LearnW.get_indices(w_range, w_trials)

        for a in range(n_arms):
            results["learn_transitionerrors"][t, a] = np.max(np.abs(est_transitions[:, :, :, a] - true_dyn[:, :, :, a]))
            results["learn_indexerrors"][t, a] = np.max(np.abs(LearnW.whittle_indices[a] - PlanW.whittle_indices[a]))
            results["plan_rewards"][t, a] = plan_totalrewards[a]
            results["plan_objectives"][t, a] = compute_utility(plan_totalrewards[a], threshold, u_type, u_order)
            results["learn_rewards"][t, a] = learn_totalrewards[a]
            results["learn_objectives"][t, a] = compute_utility(learn_totalrewards[a], threshold, u_type, u_order)

    print(f"Iteration {i} end with duration: {time.time() - start_time}")
    return results


def multiprocess_inf_learn_LRAPTS(
        n_iterations, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
        true_rew, true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    PlanW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, true_dyn, discount, n_steps, u_type, u_order, threshold)
    PlanW.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials) 
        for i in range(n_iterations)
    ]

    # Use multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.starmap(process_inf_learn_LRAPTS_iteration, args)

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
