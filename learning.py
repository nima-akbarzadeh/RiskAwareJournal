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
        (i, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials) 
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


def process_inf_learn_LRAPTS_iteration(i, l_episodes, n_episodes, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
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
    LearnW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, est_transitions, discount, n_steps, u_type, u_order, threshold)
    LearnW.get_indices(w_range, w_trials)
    counts = np.ones((n_states, n_states, 2, n_arms))

    for l in range(l_episodes):
        plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
            process_inf_riskaware_whittle_learning(PlanW, LearnW, n_episodes, discount, n_steps, n_states, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order)
        counts += cnts
        
        # Update transitions
        est_transitions = np.zeros((n_states, n_states, 2, n_arms))
        for a in range(n_arms):
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
        LearnW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, est_transitions, discount, n_steps, u_type, u_order, threshold)
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


def multiprocess_inf_learn_LRAPTS(
        n_iterations, l_episodes, n_episodes, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, 
        true_rew, true_dyn, initial_states, u_type, u_order, save_data, filename, w_range, w_trials
        ):
    num_workers = cpu_count() - 1

    PlanW = RiskAwareWhittleInf([n_states, n_augmnts, n_augmnts], n_arms, true_rew, true_dyn, discount, n_steps, u_type, u_order, threshold)
    PlanW.get_indices(w_range, w_trials)

    # Define arguments for each iteration
    args = [
        (i, l_episodes, n_episodes, discount, n_steps, n_states, n_augmnts, n_arms, n_choices, threshold, true_rew, true_dyn, initial_states, u_type, u_order, PlanW, w_range, w_trials) 
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
