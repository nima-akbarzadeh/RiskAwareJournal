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
        vals[x, :] = np.round(per_step_reward / time_horizon, 3) * np.ones(num_arms)
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
    
    def normalize_transition_row(row):
        """Ensure a transition row sums to 1 and has non-negative values."""
        # Clip negative values to 0
        row = np.maximum(row, 0)
        # Normalize to sum to 1
        row_sum = np.sum(row)
        if row_sum > 0:
            row = row / row_sum
        else:
            # If all zeros, set to uniform distribution
            row = np.ones_like(row) / len(row)
        return row
    
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
            transitions[1:, 0, 0, a] = 1 - (num_states - 1) * p
            
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

    elif transition_type == 'clinical-v2':
        for a in range(num_arms):
            # Extract probabilities
            pr_ss_0 = prob_remain[0][a]
            pr_sr_0 = prob_remain[1][a]
            pr_sp_0 = prob_remain[2][a]
            pr_rr_0 = prob_remain[3][a]
            pr_rp_0 = prob_remain[4][a]
            pr_pp_0 = prob_remain[5][a]

            pr_ss_1 = prob_remain[6][a]
            pr_sr_1 = prob_remain[7][a]
            pr_sp_1 = prob_remain[8][a]
            pr_rr_1 = prob_remain[9][a] 
            pr_rp_1 = prob_remain[10][a]
            pr_pp_1 = prob_remain[11][a]

            # For action 0
            # State P (index 1): transitions to D or P
            trans_p_0 = np.array([1 - pr_pp_0, pr_pp_0, 0, 0])
            
            # State R (index 2): transitions to D, P, or R
            trans_r_0 = np.array([0, pr_rp_0, pr_rr_0, 0])
            # Calculate transition to D as remainder, ensuring non-negative
            trans_r_0[0] = max(0, 1 - (pr_rp_0 + pr_rr_0))
            
            # State S (index 3): transitions to D, P, R, or S
            trans_s_0 = np.array([0, pr_sp_0, pr_sr_0, pr_ss_0])
            # Calculate transition to D as remainder, ensuring non-negative
            trans_s_0[0] = max(0, 1 - (pr_sp_0 + pr_sr_0 + pr_ss_0))
            
            # Normalize rows to ensure they sum to 1
            trans_p_0 = normalize_transition_row(trans_p_0)
            trans_r_0 = normalize_transition_row(trans_r_0)
            trans_s_0 = normalize_transition_row(trans_s_0)
            
            transitions[:, :, 0, a] = np.array([
                [1, 0, 0, 0],  # State D (Death) - absorbing
                trans_p_0,     # From state P
                trans_r_0,     # From state R
                trans_s_0      # From state S
            ])
            
            # For action 1
            # State P (index 1): transitions to D or P
            trans_p_1 = np.array([1 - pr_pp_1, pr_pp_1, 0, 0])
            
            # State R (index 2): transitions to D, P, or R
            trans_r_1 = np.array([0, pr_rp_1, pr_rr_1, 0])
            # Calculate transition to D as remainder, ensuring non-negative
            trans_r_1[0] = max(0, 1 - (pr_rp_1 + pr_rr_1))
            
            # State S (index 3): transitions to D, P, R, or S
            trans_s_1 = np.array([0, pr_sp_1, pr_sr_1, pr_ss_1])
            # Calculate transition to D as remainder, ensuring non-negative
            trans_s_1[0] = max(0, 1 - (pr_sp_1 + pr_sr_1 + pr_ss_1))
            
            # Normalize rows to ensure they sum to 1
            trans_p_1 = normalize_transition_row(trans_p_1)
            trans_r_1 = normalize_transition_row(trans_r_1)
            trans_s_1 = normalize_transition_row(trans_s_1)
            
            transitions[:, :, 1, a] = np.array([
                [1, 0, 0, 0],  # State D (Death) - absorbing
                trans_p_1,     # From state P
                trans_r_1,     # From state R
                trans_s_1      # From state S
            ])

    elif transition_type == 'clinical-v3':
        for a in range(num_arms):
            # Extract probabilities
            pr_ss_0 = prob_remain[0][a]
            pr_sr_0 = prob_remain[1][a]
            pr_rr_0 = prob_remain[2][a]
            pr_pp_0 = prob_remain[3][a]

            pr_ss_1 = prob_remain[4][a]
            pr_sr_1 = prob_remain[5][a]
            pr_rr_1 = prob_remain[6][a]
            pr_pp_1 = prob_remain[7][a]

            # For action 0
            trans_p_0 = np.array([1 - pr_pp_0, pr_pp_0, 0, 0])
            trans_r_0 = np.array([0, 1 - pr_rr_0, pr_rr_0, 0])
            trans_s_0 = np.array([0, 1 - (pr_sr_0 + pr_ss_0), pr_sr_0, pr_ss_0])
            
            # Normalize rows
            trans_p_0 = normalize_transition_row(trans_p_0)
            trans_r_0 = normalize_transition_row(trans_r_0)
            trans_s_0 = normalize_transition_row(trans_s_0)
            
            transitions[:, :, 0, a] = np.array([
                [1, 0, 0, 0],  # State D (Death)
                trans_p_0,     # From state P
                trans_r_0,     # From state R
                trans_s_0      # From state S
            ])
            
            # For action 1
            trans_p_1 = np.array([1 - pr_pp_1, pr_pp_1, 0, 0])
            trans_r_1 = np.array([0, 1 - pr_rr_1, pr_rr_1, 0])
            trans_s_1 = np.array([0, 1 - (pr_sr_1 + pr_ss_1), pr_sr_1, pr_ss_1])
            
            # Normalize rows
            trans_p_1 = normalize_transition_row(trans_p_1)
            trans_r_1 = normalize_transition_row(trans_r_1)
            trans_s_1 = normalize_transition_row(trans_s_1)
            
            transitions[:, :, 1, a] = np.array([
                [1, 0, 0, 0],  # State D (Death)
                trans_p_1,     # From state P
                trans_r_1,     # From state R
                trans_s_1      # From state S
            ])

    elif transition_type == 'clinical-v4':
        for a in range(num_arms):
            # Extract probabilities
            pr_ss_0 = prob_remain[0][a]
            pr_pp_0 = prob_remain[1][a]
            pr_ss_1 = prob_remain[2][a]
            pr_pp_1 = prob_remain[3][a]

            # For action 0
            trans_p_0 = np.array([1 - pr_pp_0, pr_pp_0, 0])
            trans_s_0 = np.array([0, 1 - pr_ss_0, pr_ss_0])
            
            # Normalize rows
            trans_p_0 = normalize_transition_row(trans_p_0)
            trans_s_0 = normalize_transition_row(trans_s_0)
            
            transitions[:, :, 0, a] = np.array([
                [1, 0, 0],     # State D (Death)
                trans_p_0,     # From state P
                trans_s_0      # From state S
            ])
            
            # For action 1
            trans_p_1 = np.array([1 - pr_pp_1, pr_pp_1, 0])
            trans_s_1 = np.array([0, 1 - pr_ss_1, pr_ss_1])
            
            # Normalize rows
            trans_p_1 = normalize_transition_row(trans_p_1)
            trans_s_1 = normalize_transition_row(trans_s_1)
            
            transitions[:, :, 1, a] = np.array([
                [1, 0, 0],     # State D (Death)
                trans_p_1,     # From state P
                trans_s_1      # From state S
            ])

    # Final validation: ensure all rows sum to 1 and are non-negative
    for a in range(num_arms):
        for s in range(num_states):
            for action in range(2):
                transitions[s, :, action, a] = normalize_transition_row(transitions[s, :, action, a])

    return transitions
