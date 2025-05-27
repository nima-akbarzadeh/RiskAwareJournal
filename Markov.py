### Problem Parameters
import numpy as np
from utils import compute_utility


# Define the reward values for each arm
def rewards(time_horizon, num_arms, num_states):
    """
    Calculate rewards for each arm.
    """
    vals = np.ones((num_states, num_arms))
    per_step_rewards = np.linspace(0, 1, num=num_states) 
    for x, per_step_reward in enumerate(per_step_rewards):
        vals[x, :] = np.round(per_step_reward / time_horizon, 5) * np.ones(num_arms)
    return vals

# Define the reward rewards for each arm
def rewards_utility(time_horizon, num_arms, num_states, threshold, u_type, u_order):
    """
    Calculate a utility function of the rewards for each arm.
    """
    vals = np.ones((num_states, num_arms))
    per_step_rewards = np.linspace(0, 1, num=num_states) 
    for x, per_step_reward in enumerate(per_step_rewards):
        vals[x, :] = np.round(compute_utility(per_step_reward, threshold, u_type, u_order) / time_horizon, 3) * np.ones(num_arms)
    return vals

# Define the reward rewards for each arm
def rewards_ns(discount, time_horizon, num_arms, num_states):
    """
    Calculate a utility function of the rewards for each arm.
    """
    vals = np.ones((num_states, time_horizon, num_arms))
    for t in range(time_horizon):
        per_step_rewards = (1 - discount) * (discount**t) * np.linspace(0, 1, num=num_states)  / (1 - discount ** time_horizon)
        for x, per_step_reward in enumerate(per_step_rewards):
            vals[x, t, :] = np.round(per_step_reward, 3) * np.ones(num_arms)
    return vals

# Define the reward rewards for each arm
def rewards_ns_utility(discount, time_horizon, num_arms, num_states, threshold, u_type, u_order):
    """
    Calculate a utility function of the rewards for each arm.
    """
    vals = np.ones((num_states, time_horizon, num_arms))
    for t in range(time_horizon):
        per_step_rewards = np.linspace(0, 1, num=num_states) 
        for x, per_step_reward in enumerate(per_step_rewards):
            vals[x, t, :] = np.round(
                (discount ** t) * (1 - discount) * compute_utility(per_step_reward, threshold, u_type, u_order) / (1 - discount ** time_horizon)
            , 3) * np.ones(num_arms)
    return vals

# Define the reward values for each arm
def rewards_inf(discount, time_horizon, num_arms, num_states):
    """
    Calculate rewards for each arm.
    """
    vals = np.ones((num_states, num_arms))
    per_step_rewards = (1 - discount) * np.linspace(0, 1, num=num_states)  / (1 - discount ** time_horizon)
    for x, per_step_reward in enumerate(per_step_rewards):
        vals[x, :] = np.round(per_step_reward, 3) * np.ones(num_arms)
    return vals

# Define the reward rewards for each arm
def rewards_inf_utility(discount, time_horizon, num_arms, num_states, threshold, u_type, u_order):
    """
    Calculate a utility function of the rewards for each arm.
    """
    vals = np.ones((num_states, num_arms))
    per_step_rewards = np.linspace(0, 1, num=num_states) 
    for x, per_step_reward in enumerate(per_step_rewards):
        vals[x, :] = np.round(
            (1 - discount) * compute_utility(per_step_reward, threshold, u_type, u_order) / (1 - discount ** time_horizon)
        , 3) * np.ones(num_arms)
    return vals

# Define the Markov dynamics for each arm
def get_transitions(num_arms: int, num_states: int, prob_remain: np.ndarray, 
                    transition_type: str) -> np.ndarray:
    """
    Optimized transition matrix computation with validation.
    
    Args:
        num_arms: Number of arms
        num_states: Number of states per arm
        prob_remain: Probability parameters (format depends on transition_type)
        transition_type: Either 'structured' or 'clinical'
    
    Returns:
        Transition tensor of shape (num_states, num_states, 2, num_arms)
    """
    transitions = np.zeros((num_states, num_states, 2, num_arms))
    
    if transition_type == 'structured':
        for a in range(num_arms):
            p = prob_remain[a]
            
            # Action 1 transitions (vectorized)
            for s in range(num_states - 1):
                remaining_prob = (num_states - s - 1) * p
                transitions[s, s, 1, a] = remaining_prob
                transitions[s, num_states - 1, 1, a] = 1 - remaining_prob
            transitions[num_states - 1, num_states - 1, 1, a] = 1
            
            # Action 0 transitions (vectorized where possible)
            transitions[0, 0, 0, a] = 1
            base_prob = 1 - (num_states - 1) * p
            transitions[1:, 0, 0, a] = base_prob
            
            # Fill lower triangular efficiently
            lower_tri = np.tril(np.ones((num_states - 1, num_states - 1)))
            transitions[1:, 1:, 0, a] = p * lower_tri
            
            # Diagonal correction
            for s in range(1, num_states):
                transitions[s, s, 0, a] *= (num_states - s)
    
    elif transition_type == 'clinical':
        for a in range(num_arms):
            # Extract and normalize probabilities
            probs_0 = prob_remain[:3, a] if prob_remain.ndim > 1 else prob_remain[:3]
            probs_1 = prob_remain[3:6, a] if prob_remain.ndim > 1 else prob_remain[3:6]
            
            # Normalize if needed
            pr_ss_0, pr_sp_0, pr_pp_0 = probs_0
            if pr_ss_0 + pr_sp_0 > 1:
                total = pr_ss_0 + pr_sp_0
                pr_ss_0 = np.round(pr_ss_0 / total, 3)
                pr_sp_0 = np.round(pr_sp_0 / total, 3)
            
            pr_ss_1, pr_sp_1, pr_pp_1 = probs_1
            if pr_ss_1 + pr_sp_1 > 1:
                total = pr_ss_1 + pr_sp_1
                pr_ss_1 = np.round(pr_ss_1 / total, 3)
                pr_sp_1 = np.round(pr_sp_1 / total, 3)
            
            # Set transition matrices
            transitions[:, :, 0, a] = np.array([
                [1, 0, 0],
                [1 - pr_pp_0, pr_pp_0, 0],
                [1 - (pr_sp_0 + pr_ss_0), pr_sp_0, pr_ss_0]
            ])
            
            transitions[:, :, 1, a] = np.array([
                [1, 0, 0],
                [1 - pr_pp_1, pr_pp_1, 0],
                [1 - (pr_sp_1 + pr_ss_1), pr_sp_1, pr_ss_1]
            ])
    
    # Validate and normalize transition probabilities
    for a in range(num_arms):
        for s in range(num_states):
            for action in range(2):
                prob_sum = np.sum(transitions[s, :, action, a])
                if not np.isclose(prob_sum, 1.0, atol=1e-6):
                    if prob_sum > 0:
                        transitions[s, :, action, a] /= prob_sum
                    else:
                        # Default: stay in same state
                        transitions[s, s, action, a] = 1.0
    
    return transitions
