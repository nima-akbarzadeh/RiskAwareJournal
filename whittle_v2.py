import numpy as np
from itertools import product, combinations
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Callable
import warnings
from processes import compute_utility

# ============================================================================
# Utility Functions
# ============================================================================

def possible_reward_sums_dp(rewards: np.ndarray, num_steps: int) -> List[float]:
    """
    Compute possible reward sums using dynamic programming to avoid 
    exponential memory usage with itertools.product.
    """
    if num_steps == 0:
        return [0.0]
    
    # Use a set to track unique sums at each step
    current_sums = {0.0}
    unique_rewards = np.unique(np.round(rewards, 3))
    
    for _ in range(num_steps):
        next_sums = set()
        for current_sum in current_sums:
            for reward in unique_rewards:
                next_sums.add(np.round(current_sum + reward, 3))
        current_sums = next_sums
    
    return sorted(list(current_sums))


def possible_reward_sums(rewards: np.ndarray, num_steps: int) -> List[float]:
    """Original function kept for compatibility."""
    if num_steps <= 10:  # Use original for small cases
        reward_combinations = product(rewards, repeat=num_steps)
        sums = set()
        for combination in reward_combinations:
            sums.add(np.round(sum(combination), 3))
        return sorted(sums)
    else:  # Use DP for larger cases
        return possible_reward_sums_dp(rewards, num_steps)


# ============================================================================
# Static Binary Search Algorithm
# ============================================================================

def compute_indices(num_arms: int,
                    backward_func: Callable,
                    indexability_check_func: Callable,
                    policy_equal_func: Callable,
                    index_range: float,
                    n_trials: int) -> List[np.ndarray]:
    """
    Generic binary search for Whittle indices.
    
    Args:
        num_arms: Number of arms
        backward_func: Function(arm, penalty) -> (policy, value, Q)
        indexability_check_func: Function(arm_indices, nxt_pol, ref_pol, penalty, arm) -> (bool, indices)
        policy_equal_func: Function(pol1, pol2) -> bool
        index_range: Upper bound for search
        n_trials: Number of trials for granularity
        
    Returns:
        List of Whittle indices for each arm
    """
    l_steps = index_range / n_trials
    whittle_indices = []
    
    for arm in range(num_arms):
        # Get initial shape from a dummy backward call
        ref_pol, _, ref_q = backward_func(arm, 0)
        arm_indices = np.zeros_like(ref_pol, dtype=float)
        
        penalty_ref = 0
        ref_pol, _, ref_q = backward_func(arm, penalty_ref)
        ubp_pol, _, _ = backward_func(arm, index_range)
        
        if policy_equal_func(ref_pol, ubp_pol):
            whittle_indices.append(arm_indices)
            continue
        
        while not policy_equal_func(ref_pol, ubp_pol):
            lb_temp = penalty_ref
            ub_temp = index_range
            
            # Binary search for policy change point
            while (ub_temp - lb_temp) > l_steps:
                penalty = 0.5 * (lb_temp + ub_temp)
                som_pol, _, _ = backward_func(arm, penalty)
                
                if policy_equal_func(som_pol, ref_pol):
                    lb_temp = penalty
                else:
                    ub_temp = penalty
            
            penalty_ref = lb_temp + l_steps
            nxt_pol, _, nxt_q = backward_func(arm, penalty_ref)
            
            flag, arm_indices = indexability_check_func(
                arm_indices, nxt_pol, ref_pol, nxt_q, ref_q, lb_temp, arm
            )
            
            if flag:
                ref_pol = nxt_pol.copy()
                ref_q = nxt_q.copy()
            else:
                break
        
        whittle_indices.append(arm_indices)
    
    return whittle_indices


# ============================================================================
# Base Classes
# ============================================================================

class BaseWhittle(ABC):
    """Base class for all Whittle index computations."""
    
    def __init__(self, num_states: int, num_arms: int, transition: np.ndarray,
                 horizon: int, digits: int = 3):
        self.num_x = num_states
        self.num_a = num_arms
        self.transition = transition
        self.horizon = horizon
        self.digits = digits
        self.whittle_indices = []
        self.penalty_factor = 1.0 / self.horizon if self.horizon > 0 else 0
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input dimensions and probabilities."""
        expected_shape = (self.num_x, self.num_x, 2, self.num_a)
        if self.transition.shape != expected_shape:
            raise ValueError(f"Transition shape mismatch. Expected {expected_shape}, got {self.transition.shape}")
        
        # Validate transition probabilities
        for a in range(self.num_a):
            for x in range(self.num_x):
                for action in range(2):
                    prob_sum = np.sum(self.transition[x, :, action, a])
                    if not np.isclose(prob_sum, 1.0, atol=1e-6):
                        warnings.warn(f"Transition probabilities sum to {prob_sum} for state {x}, action {action}, arm {a}")
    
    def get_indices(self, index_range: float, n_trials: int):
        """Compute Whittle indices using static binary search."""
        self.whittle_indices = compute_indices(
            num_arms=self.num_a,
            backward_func=self.backward,
            indexability_check_func=self.indexability_check,
            policy_equal_func=self.policy_equal,
            index_range=index_range,
            n_trials=n_trials
        )
    
    @abstractmethod
    def backward(self, arm: int, penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward induction for computing optimal policy."""
        pass
    
    @abstractmethod
    def indexability_check(self, arm_indices: np.ndarray, nxt_pol: np.ndarray,
                          ref_pol: np.ndarray, nxt_q: np.ndarray,
                          ref_q: np.ndarray, penalty: float, arm: int) -> Tuple[bool, np.ndarray]:
        """Check indexability condition."""
        pass
    
    @abstractmethod
    def policy_equal(self, pol1: np.ndarray, pol2: np.ndarray) -> bool:
        """Check if two policies are equal."""
        pass
    
    @abstractmethod
    def take_action(self, n_choices: int, current_state: Dict) -> np.ndarray:
        """Select actions based on Whittle indices."""
        pass
    
    def _select_arms_by_indices(self, current_indices: np.ndarray, n_choices: int) -> np.ndarray:
        """Common logic for selecting arms based on Whittle indices."""
        positive_mask = current_indices >= 0
        n_choices = min(n_choices, np.sum(positive_mask))
        
        if n_choices == 0:
            return np.zeros(self.num_a, dtype=int)
        
        if n_choices < self.num_a:
            # Use argpartition for efficient top-k selection
            top_indices = np.argpartition(current_indices, -n_choices)[-n_choices:]
            valid_top = top_indices[current_indices[top_indices] >= 0]
            action_vector = np.zeros(self.num_a, dtype=int)
            action_vector[valid_top] = 1
        else:
            action_vector = positive_mask.astype(int)
        
        return action_vector


# ============================================================================
# Finite Horizon Implementations
# ============================================================================

class Whittle(BaseWhittle):
    """Risk-neutral Whittle index computation (stationary rewards)."""
    
    def __init__(self, num_states: int, num_arms: int, reward: np.ndarray,
                 transition: np.ndarray, horizon: int):
        super().__init__(num_states, num_arms, transition, horizon)
        self.reward = reward
    
    def backward(self, arm: int, penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized backward induction using vectorization."""
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)
        Q = np.zeros((self.num_x, self.horizon, 2), dtype=np.float32)
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)
        
        penalty_term = penalty * self.penalty_factor
        
        for t in range(self.horizon - 1, -1, -1):
            # Vectorized Q-value computation
            Q[:, t, 0] = self.reward[:, arm] + self.transition[:, :, 0, arm] @ V[:, t + 1]
            Q[:, t, 1] = self.reward[:, arm] - penalty_term + self.transition[:, :, 1, arm] @ V[:, t + 1]
            
            # Optimal policy and value
            pi[:, t] = np.argmax(Q[:, t, :], axis=1)
            V[:, t] = np.max(Q[:, t, :], axis=1)
        
        return pi, V, Q
    
    def indexability_check(self, arm_indices: np.ndarray, nxt_pol: np.ndarray, 
                           ref_pol: np.ndarray, nxt_q: np.ndarray,
                          ref_q: np.ndarray, penalty: float, arm: int) -> Tuple[bool, np.ndarray]:
        """Check indexability condition."""
        for t in range(self.horizon):
            violations = (ref_pol[:, t] == 0) & (nxt_pol[:, t] == 1)
            if np.any(violations):
                print(f"Neutral - Not indexable at time {t}!")
                return False, np.zeros((self.num_x, self.horizon))
            
            changes = (ref_pol[:, t] == 1) & (nxt_pol[:, t] == 0)
            arm_indices[changes, t] = penalty
        
        return True, arm_indices
    
    def policy_equal(self, pol1: np.ndarray, pol2: np.ndarray) -> bool:
        """Check if two policies are equal."""
        return np.array_equal(pol1, pol2)
    
    def take_action(self, n_choices: int, current_state: Dict) -> np.ndarray:
        """Select actions using efficient top-k selection."""
        current_x = current_state['x']
        current_t = current_state['t']
        
        current_indices = np.array([
            self.whittle_indices[arm][current_x[arm], current_t]
            for arm in range(self.num_a)
        ])
        
        return self._select_arms_by_indices(current_indices, n_choices)


class WhittleNS(Whittle):
    """Whittle index with nonstationary (time-dependent) rewards."""
    
    def __init__(self, num_states: int, num_arms: int, reward: np.ndarray,
                 transition: np.ndarray, horizon: int):
        super().__init__(num_states, num_arms, reward, transition, horizon)
        # Note: self.reward now has shape (num_states, horizon, num_arms)
    
    def backward(self, arm: int, penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward induction with time-dependent rewards."""
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)
        Q = np.zeros((self.num_x, self.horizon, 2), dtype=np.float32)
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)
        
        penalty_term = penalty * self.penalty_factor
        
        for t in range(self.horizon - 1, -1, -1):
            # Vectorized Q-value computation
            Q[:, t, 0] = self.reward[:, t, arm] + self.transition[:, :, 0, arm] @ V[:, t + 1]
            Q[:, t, 1] = self.reward[:, t, arm] - penalty_term + self.transition[:, :, 1, arm] @ V[:, t + 1]
            
            # Optimal policy and value
            pi[:, t] = np.argmax(Q[:, t, :], axis=1)
            V[:, t] = np.max(Q[:, t, :], axis=1)
        
        return pi, V, Q


class RiskAwareWhittle(BaseWhittle):
    """Risk-aware Whittle with augmented state space."""
    
    def __init__(self, num_states: int, num_arms: int, rewards: np.ndarray,
                 transition: np.ndarray, horizon: int, u_type: str,
                 u_order: float, threshold: float):
        super().__init__(num_states, num_arms, transition, horizon)
        self.rewards = rewards
        self.u_type = u_type
        self.u_order = u_order
        self.threshold = threshold
        
        self._initialize_augmented_space()
    
    def _initialize_augmented_space(self):
        """Initialize augmented state space efficiently."""
        self.n_realize = []
        self.n_augment = []
        self.all_rews = []
        self.all_utility_values = []
        
        for a in range(self.num_a):
            unique_rewards = np.unique(self.rewards[:, a])
            arm_n_realize = []
            
            # Use DP for large horizons
            for t in range(self.horizon):
                if t < 10:
                    all_total_rewards_by_t = possible_reward_sums(unique_rewards, t + 1)
                else:
                    all_total_rewards_by_t = possible_reward_sums_dp(unique_rewards, t + 1)
                
                arm_n_realize.append(len(all_total_rewards_by_t))
                if t == self.horizon - 1:
                    all_total_rewards = all_total_rewards_by_t
            
            self.n_augment.append(len(all_total_rewards))
            self.all_rews.append(all_total_rewards)
            self.n_realize.append(arm_n_realize)
            
            # Compute utilities
            arm_utilities = np.array([
                compute_utility(r, self.threshold, self.u_type, self.u_order)
                for r in all_total_rewards
            ])
            self.all_utility_values.append(arm_utilities)
    
    def backward(self, arm: int, penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward induction with lifted state space."""
        
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)
        
        # Terminal values
        V[:, :, self.horizon] = self.all_utility_values[arm][:, np.newaxis]
        
        penalty_term = penalty * self.penalty_factor
        
        for t in range(self.horizon - 1, -1, -1):
            n_real = self.n_realize[arm][t]
            
            for l in range(n_real):
                for x in range(self.num_x):
                    nxt_l = np.clip(l + x, 0, self.n_augment[arm] - 1)
                    
                    # Q-values using matrix multiplication
                    Q[l, x, t, 0] = self.transition[x, :, 0, arm] @ V[nxt_l, :, t + 1]
                    Q[l, x, t, 1] = -penalty_term + self.transition[x, :, 1, arm] @ V[nxt_l, :, t + 1]
            
            # Optimal policy and value
            pi[:n_real, :, t] = np.argmax(Q[:n_real, :, t, :], axis=2)
            V[:n_real, :, t] = np.max(Q[:n_real, :, t, :], axis=2)
        
        return pi, V, Q
    
    def indexability_check(self, arm_indices: np.ndarray, nxt_pol: np.ndarray,
                          ref_pol: np.ndarray, nxt_q: np.ndarray,
                          ref_q: np.ndarray, penalty: float, arm: int) -> Tuple[bool, np.ndarray]:
        """Check indexability for risk-aware setting."""
        realize_index = self.n_realize[arm]
        
        for t in range(self.horizon):
            n_real = realize_index[t]
            ref_pol_t = ref_pol[:n_real, :, t]
            nxt_pol_t = nxt_pol[:n_real, :, t]
            
            violations = (ref_pol_t == 0) & (nxt_pol_t == 1)
            if np.any(violations):
                print(f"RA - Not indexable at time {t}!")
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            
            changes = (ref_pol_t == 1) & (nxt_pol_t == 0)
            change_indices = np.argwhere(changes)
            arm_indices[change_indices[:, 0], change_indices[:, 1], t] = penalty
        
        return True, arm_indices
    
    def policy_equal(self, pol1: np.ndarray, pol2: np.ndarray) -> bool:
        """Check if two policies are equal for realized states."""
        realize_index = self.n_realize[self._current_arm]  # Note: we need to track current arm
        
        for t in range(self.horizon):
            n_real = realize_index[t]
            if not np.array_equal(pol1[:n_real, :, t], pol2[:n_real, :, t]):
                return False
        return True
    
    def get_indices(self, index_range: float, n_trials: int):
        """Override to handle arm-specific policy comparison."""
        self.whittle_indices = []
        
        for arm in range(self.num_a):
            self._current_arm = arm  # Track current arm for policy_equal
            
            indices = compute_indices(
                num_arms=1,  # Process one arm at a time
                backward_func=lambda a, p: self.backward(arm, p),
                indexability_check_func=lambda idx, nxt, ref, nxt_q, ref_q, pen, a: self.indexability_check(idx, nxt, ref, nxt_q, ref_q, pen, arm),
                policy_equal_func=self.policy_equal,
                index_range=index_range,
                n_trials=n_trials
            )
            self.whittle_indices.append(indices[0])
    
    def take_action(self, n_choices: int, current_state: Dict) -> np.ndarray:
        """Select actions for risk-aware setting."""
        current_l = current_state['l']
        current_x = current_state['x']
        current_t = current_state['t']
        
        current_indices = np.array([
            self.whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            for arm in range(self.num_a)
        ])
        
        return self._select_arms_by_indices(current_indices, n_choices)


class RiskAwareWhittleNS(BaseWhittle):
    """Risk-aware Whittle with nonstationary rewards and discretized augmented space."""
    
    def __init__(self, num_states: Tuple[int, int], num_arms: int, rewards: np.ndarray,
                 transition: np.ndarray, horizon: int, u_type: str,
                 u_order: float, threshold: float):
        # Handle tuple input for num_states
        self.num_x = num_states[0]
        self.num_s = num_states[1]
        
        # Initialize parent with physical states only
        super().__init__(self.num_x, num_arms, transition, horizon)
        
        self.rewards = rewards  # Time-dependent rewards
        self.u_type = u_type
        self.u_order = u_order
        self.threshold = threshold
        
        # Discretize the augmented state space
        self.cutting_points = np.round(np.linspace(0, horizon, self.num_s + 1), 2)
        self.all_total_rewards = np.round([
            np.median(self.cutting_points[i:i + 2])
            for i in range(len(self.cutting_points) - 1)
        ], 2)
        self.n_augment = len(self.all_total_rewards)
        
        # Compute utilities
        self.all_utility_values = np.array([
            compute_utility(r, threshold, u_type, u_order)
            for r in self.all_total_rewards
        ])
    
    def get_reward_partition(self, reward_value: float) -> int:
        """Find which partition a reward value belongs to."""
        index = np.searchsorted(self.cutting_points, reward_value, side='right')
        return min(index - 1, len(self.cutting_points) - 2)
    
    def backward(self, arm: int, penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward induction with discretized augmented state."""
        V = np.zeros((self.n_augment, self.num_x, self.horizon + 1), dtype=np.float32)
        Q = np.zeros((self.n_augment, self.num_x, self.horizon, 2), dtype=np.float32)
        pi = np.zeros((self.n_augment, self.num_x, self.horizon), dtype=np.int32)
        
        # Terminal values
        V[:, :, self.horizon] = self.all_utility_values[:, np.newaxis]
        
        penalty_term = penalty * self.penalty_factor
        
        for t in range(self.horizon - 1, -1, -1):
            for x in range(self.num_x):
                for l in range(self.n_augment):
                    # Next augmented state
                    nxt_l = self.get_reward_partition(
                        self.all_total_rewards[l] + self.rewards[x, t, arm]
                    )
                    
                    # Q-values
                    Q[l, x, t, 0] = self.transition[x, :, 0, arm] @ V[nxt_l, :, t + 1]
                    Q[l, x, t, 1] = -penalty_term + self.transition[x, :, 1, arm] @ V[nxt_l, :, t + 1]
                    
                    # Optimal policy and value
                    if Q[l, x, t, 0] < Q[l, x, t, 1]:
                        pi[l, x, t] = 1
                        V[l, x, t] = Q[l, x, t, 1]
                    else:
                        pi[l, x, t] = 0
                        V[l, x, t] = Q[l, x, t, 0]

                    pi[l, x, t] = np.argmax(Q[l, x, t, :])
                    V[l, x, t] = np.max(Q[l, x, t, :])
        
        return pi, V, Q
    
    def indexability_check(self, arm_indices: np.ndarray, nxt_pol: np.ndarray,
                          ref_pol: np.ndarray, nxt_q: np.ndarray,
                          ref_q: np.ndarray, penalty: float, arm: int) -> Tuple[bool, np.ndarray]:
        """Check indexability for discretized augmented state."""
        for t in range(self.horizon):
            ref_pol_t = ref_pol[:, :, t]
            nxt_pol_t = nxt_pol[:, :, t]
            
            violations = (ref_pol_t == 0) & (nxt_pol_t == 1)
            if np.any(violations):
                print(f"RA - Not indexable at time {t}!")
                return False, np.zeros((self.n_augment, self.num_x, self.horizon))
            
            changes = (ref_pol_t == 1) & (nxt_pol_t == 0)
            change_indices = np.argwhere(changes)
            arm_indices[change_indices[:, 0], change_indices[:, 1], t] = penalty
        
        return True, arm_indices
    
    def policy_equal(self, pol1: np.ndarray, pol2: np.ndarray) -> bool:
        """Check matrix equality."""
        return np.array_equal(pol1, pol2)
    
    def take_action(self, n_choices: int, current_state: Dict) -> np.ndarray:
        """Select actions for risk-aware nonstationary setting."""
        current_l = current_state['l']
        current_x = current_state['x']
        current_t = current_state['t']
        
        current_indices = np.array([
            self.whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            for arm in range(self.num_a)
        ])
        
        return self._select_arms_by_indices(current_indices, n_choices)


# ============================================================================
# Infinite Horizon Implementations
# ============================================================================

class BaseWhittleInf(ABC):
    """Base class for infinite horizon Whittle indices."""
    
    def __init__(self, num_states: int, num_arms: int, transition: np.ndarray,
                 horizon: int, discount: float = 1, digits: int = 3):
        self.num_x = num_states
        self.num_a = num_arms
        self.transition = transition
        self.horizon = horizon  # Max iterations for value iteration
        self.discount = discount
        self.digits = digits
        self.whittle_indices = []
        self.convergence_tol = 1e-4
    
    def get_indices(self, index_range: float, n_trials: int):
        """Compute Whittle indices using static binary search."""
        self.whittle_indices = compute_indices(
            num_arms=self.num_a,
            backward_func=self.bellman,  # Note: bellman is essentially backward for infinite horizon
            indexability_check_func=self.indexability_check,
            policy_equal_func=self.policy_equal,
            index_range=index_range,
            n_trials=n_trials
        )
    
    @abstractmethod
    def bellman(self, arm: int, penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Value iteration (backward induction for infinite horizon)."""
        pass
    
    @abstractmethod
    def indexability_check(self, arm_indices: np.ndarray, nxt_pol: np.ndarray, 
                           ref_pol: np.ndarray, nxt_q: np.ndarray, 
                           ref_q: np.ndarray, penalty: float, arm: int) -> Tuple[bool, np.ndarray]:
        """Check indexability condition."""
        pass
    
    @abstractmethod
    def policy_equal(self, pol1: np.ndarray, pol2: np.ndarray) -> bool:
        """Check if two policies are equal."""
        pass
    
    @abstractmethod
    def take_action(self, n_choices: int, current_state: Dict) -> np.ndarray:
        """Select actions based on Whittle indices."""
        pass
    
    def _select_arms_by_indices(self, current_indices: np.ndarray, n_choices: int) -> np.ndarray:
        """Common logic for selecting arms based on Whittle indices."""
        positive_mask = current_indices >= 0
        n_choices = min(n_choices, np.sum(positive_mask))
        
        if n_choices == 0:
            return np.zeros(self.num_a, dtype=int)
        
        if n_choices < self.num_a:
            top_indices = np.argpartition(current_indices, -n_choices)[-n_choices:]
            valid_top = top_indices[current_indices[top_indices] >= 0]
            action_vector = np.zeros(self.num_a, dtype=int)
            action_vector[valid_top] = 1
        else:
            action_vector = positive_mask.astype(int)
        
        return action_vector


class WhittleInf(BaseWhittleInf):
    """Infinite horizon Whittle index using value iteration."""
    
    def __init__(self, num_states: int, num_arms: int, reward: np.ndarray,
                 transition: np.ndarray, horizon: int, discount: float = 1):
        super().__init__(num_states, num_arms, transition, horizon, discount)
        self.reward = reward
    
    def bellman(self, arm: int, penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Value iteration with Bellman operator for average reward (undiscounted) problems."""
        V = np.zeros(self.num_x, dtype=np.float32)
        Q = np.zeros((self.num_x, 2), dtype=np.float32)
        pi = np.zeros(self.num_x, dtype=np.int32)

        if self.discount == 1:

            penalty_term = penalty
            
            for iteration in range(self.horizon):
                v_prev = V.copy()
                
                # Vectorized Q-value computation for average reward
                # Q(s,a) = r(s,a) - rho + sum_s' P(s'|s,a) * V(s')
                Q[:, 0] = self.reward[:, arm] - rho + (self.transition[:, :, 0, arm] @ V)
                Q[:, 1] = self.reward[:, arm] - penalty_term - rho + (self.transition[:, :, 1, arm] @ V)
                
                # Optimal policy and value
                for x in range(self.num_x):
                    if Q[x, 0] < Q[x, 1]:
                        pi[x] = 1
                        V[x] = Q[x, 1]
                    else:
                        pi[x] = 0
                        V[x] = Q[x, 0]
                
                # Update average reward estimate using the Bellman equation
                # For average reward: rho = max_a [r(s,a) + sum_s' P(s'|s,a) * (V(s') - V(s))]
                # We can estimate rho from the difference in value functions
                rho = np.mean(V - v_prev)
                
                # Alternative: Use policy evaluation to estimate average reward
                # This is more stable but requires solving for steady-state
                if iteration > 10:  # Allow some iterations for V to stabilize
                    # Estimate steady-state distribution under current policy
                    pi_transitions = np.zeros((self.num_x, self.num_x))
                    pi_rewards = np.zeros(self.num_x)
                    
                    for s in range(self.num_x):
                        action = pi[s]
                        pi_transitions[s, :] = self.transition[s, :, action, arm]
                        pi_rewards[s] = self.reward[s, arm]
                        if action == 1:  # Active action
                            pi_rewards[s] -= penalty
                    
                    # Solve for steady-state: π * P = π, sum π = 1
                    try:
                        # Create matrix (P^T - I) with additional constraint sum π = 1
                        A = pi_transitions.T - np.eye(self.num_x)
                        A = np.vstack([A, np.ones(self.num_x)])
                        b = np.zeros(self.num_x + 1)
                        b[-1] = 1
                        
                        # Solve least squares (overdetermined system)
                        steady_state, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        
                        if np.all(steady_state >= 0) and np.abs(np.sum(steady_state) - 1) < 1e-6:
                            rho = np.dot(steady_state, pi_rewards)
                    except np.linalg.LinAlgError:
                        # Fall back to simple estimate if steady-state computation fails
                        pass
                
                # Check convergence for both value function and average reward
                v_conv = np.max(np.abs(V - v_prev)) < self.convergence_tol
                rho_conv = abs(rho - rho_prev) < self.convergence_tol
                
                if v_conv and rho_conv and iteration > 20:  # Ensure sufficient iterations
                    break
            
            return pi, V, Q
    
        else:
        
            penalty_term = (1 - self.discount) * penalty
            
            for iteration in range(self.horizon):
                v_prev = V.copy()
                
                # Vectorized Q-value computation
                Q[:, 0] = self.reward[:, arm] + self.discount * (self.transition[:, :, 0, arm] @ V)
                Q[:, 1] = self.reward[:, arm] - penalty_term + self.discount * (self.transition[:, :, 1, arm] @ V)
                
                # Optimal policy and value
                for x in range(self.num_x):
                    if Q[x, 0] < Q[x, 1]:
                        pi[x] = 1
                        V[x] = Q[x, 1]
                    else:
                        pi[x] = 0
                        V[x] = Q[x, 0]
                
                # Check convergence
                if np.max(np.abs(V - v_prev)) < self.convergence_tol:
                    break
            
            return pi, V, Q
    
    def indexability_check(self, arm_indices: np.ndarray, nxt_pol: np.ndarray, 
                           ref_pol: np.ndarray, nxt_q: np.ndarray,
                          ref_q: np.ndarray, penalty: float, arm: int) -> Tuple[bool, np.ndarray]:
        """Check indexability for infinite horizon."""
        violations = (ref_pol == 0) & (nxt_pol == 1)
        if np.any(violations):
            print("Neutral - Not indexable!")
            return False, np.zeros(self.num_x)
        
        changes = (ref_pol == 1) & (nxt_pol == 0)
        arm_indices[changes] = penalty
        
        return True, arm_indices
    
    def policy_equal(self, pol1: np.ndarray, pol2: np.ndarray) -> bool:
        """Check if two policies are equal."""
        return np.array_equal(pol1, pol2)
    
    def take_action(self, n_choices: int, current_state: Dict) -> np.ndarray:
        """Select actions for infinite horizon."""
        current_x = current_state['x']
        
        current_indices = np.array([
            self.whittle_indices[arm][current_x[arm]]
            for arm in range(self.num_a)
        ])
        
        return self._select_arms_by_indices(current_indices, n_choices)


class RiskAwareWhittleInf(BaseWhittleInf):
    """Risk-aware infinite horizon with discretized augmented state."""
    
    def __init__(self, num_states: Tuple[int, int, int], num_arms: int,
                 rewards: np.ndarray, transition: np.ndarray, discount: float,
                 u_type: str, u_order: float, threshold: float):
        self.num_x = num_states[0]  # Physical states
        self.num_s = num_states[1]  # Discretization of augmented state
        self.num_t = num_states[2]  # Horizon for backward induction
        
        super().__init__(self.num_x, num_arms, transition, self.num_t, discount)
        
        self.rewards = rewards
        self.u_type = u_type
        self.u_order = u_order
        self.threshold = threshold
        
        # Discretize augmented state space
        self.s_cutting_points = np.linspace(0, 1, self.num_s + 1)
        self.all_total_rewards = np.round([
            np.median(self.s_cutting_points[i:i + 2])
            for i in range(len(self.s_cutting_points) - 1)
        ], 3)
        self.n_augment = len(self.all_total_rewards)
        
        # Compute utilities
        self.all_utility_values = np.array([
            compute_utility(r, threshold, u_type, u_order)
            for r in self.all_total_rewards
        ])
    
    def get_reward_partition(self, reward_value: float) -> int:
        """Find partition for reward value."""
        index = np.searchsorted(self.s_cutting_points, reward_value, side='right')
        return min(index - 1, len(self.s_cutting_points) - 2)
    
    def bellman(self, arm: int, penalty: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward induction for infinite horizon risk-aware setting."""
        V = np.zeros((self.n_augment, self.num_x, self.num_t + 1), dtype=np.float32)
        Q = np.zeros((self.n_augment, self.num_x, self.num_t, 2), dtype=np.float32)
        pi = np.zeros((self.n_augment, self.num_x, self.num_t), dtype=np.int32)
        
        # Terminal values
        V[:, :, self.num_t] = self.all_utility_values[:, np.newaxis]
        
        penalty_term = penalty / self.num_t
        
        # Backward induction
        for t in range(self.num_t - 1, -1, -1):
            discount_factor = self.discount ** t
            
            for x in range(self.num_x):
                for y in range(self.n_augment):
                    # Next augmented state
                    reward_increment = discount_factor * self.rewards[x, arm]
                    nxt_y = self.get_reward_partition(
                        self.all_total_rewards[y] + reward_increment
                    )
                    
                    # Q-values
                    Q[y, x, t, 0] = np.round(self.transition[x, :, 0, arm] @ V[nxt_y, :, t + 1], 3)
                    Q[y, x, t, 1] = np.round(-penalty_term + self.transition[x, :, 1, arm] @ V[nxt_y, :, t + 1], 3)
                    
                    # Optimal policy and value
                    if Q[y, x, t, 0] < Q[y, x, t, 1]:
                        pi[y, x, t] = Q[y, x, t, 1]
                        V[y, x, t] = Q[y, x, t, 1]
                    else:
                        pi[y, x, t] = Q[y, x, t, 0]
                        V[y, x, t] = Q[y, x, t, 0]

        
        return pi, V, Q
    
    def indexability_check(self, arm_indices: np.ndarray, nxt_pol: np.ndarray, 
                           ref_pol: np.ndarray, nxt_q: np.ndarray, 
                           ref_q: np.ndarray, penalty: float, arm: int) -> Tuple[bool, np.ndarray]:
        """Check indexability for risk-aware infinite horizon."""
        violations = (ref_pol == 0) & (nxt_pol == 1)
        if np.any(violations):
            print("RA - Not indexable!")
            violation_indices = np.where(violations)
            
            print("Violating states (arm_indices, policy_next, policy_ref, q_next, q_ref):")
            for i in range(len(violation_indices[0])):
                current_indices = tuple(dim_idx[i] for dim_idx in violation_indices)
                print(f"  State Index: {current_indices}")
                print(f"    arm_index: {arm_indices[current_indices]}")
                print(f"    pi_next: {nxt_pol[current_indices]}")
                print(f"    pi_ref: {ref_pol[current_indices]}")
                print(f"    q_next: {nxt_q[current_indices]}")
                print(f"    q_ref: {ref_q[current_indices]}")

            return False, np.zeros((self.n_augment, self.num_x, self.num_t))
        
        changes = (ref_pol == 1) & (nxt_pol == 0)
        change_indices = np.argwhere(changes)
        arm_indices[change_indices[:, 0], change_indices[:, 1], change_indices[:, 2]] = penalty
        
        return True, arm_indices
    
    def policy_equal(self, pol1: np.ndarray, pol2: np.ndarray) -> bool:
        """Check matrix equality with tolerance."""
        return np.allclose(pol1, pol2, atol=self.convergence_tol)
    
    def take_action(self, n_choices: int, current_state: Dict) -> np.ndarray:
        """Select actions for risk-aware infinite horizon."""
        current_l = current_state['l']
        current_x = current_state['x']
        current_t = current_state['t']
        
        current_indices = np.array([
            self.whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            for arm in range(self.num_a)
        ])
        
        return self._select_arms_by_indices(current_indices, n_choices)


class OptimalRiskAwareRMAB:
    """
    Computes the optimal risk-aware policy for restless multi-armed bandits
    by solving the full joint state space problem using dynamic programming.
    """
    
    def __init__(self, num_states, num_arms, rewards, transition, discount, 
                 u_type, u_order, threshold, m_choices):
        """
        Initialize the optimal risk-aware RMAB solver.
        
        Parameters:
        - num_states: list [num_x, num_s, num_z] where:
            - num_x: number of states per arm
            - num_s: number of reward discretization buckets
            - num_z: time horizon
        - num_arms: number of arms (n)
        - rewards: array of shape (num_x, num_arms) - immediate rewards
        - transition: array of shape (num_x, num_x, 2, num_arms) - transition probabilities
        - discount: discount factor
        - u_type: utility function type
        - u_order: utility function order parameter
        - threshold: threshold for utility computation
        - m_choices: number of arms to activate at each time (m)
        """
        self.discount = discount
        self.num_x = num_states[0]
        self.num_s = num_states[1]
        self.num_z = num_states[2]
        self.num_arms = num_arms
        self.m_choices = m_choices
        self.rewards = rewards
        self.transition = transition
        self.u_type = u_type
        self.u_order = u_order
        self.threshold = threshold
        self.digits = 3
        
        # Discretize the total reward space
        self.s_cutting_points = np.linspace(0, self.num_arms, self.num_s + 1)
        self.all_total_rewards = [np.round(np.median(self.s_cutting_points[i:i + 2]), 3) 
                                  for i in range(len(self.s_cutting_points) - 1)]
        self.n_augment = len(self.all_total_rewards)
        
        # Precompute utility values
        self.all_utility_values = []
        for total_reward in self.all_total_rewards:
            self.all_utility_values.append(
                compute_utility(total_reward, threshold, u_type, u_order)
            )
        
        # Generate all possible actions (combinations of m arms from n)
        self.all_actions = list(combinations(range(num_arms), m_choices))
        self.num_actions = len(self.all_actions)
        
        # Initialize value function and policy storage
        self.V = None
        self.policy = None
        
        # Warning for large state spaces
        joint_state_size = (self.num_x ** self.num_arms) * self.n_augment * self.num_z
        if joint_state_size > 1e6:
            warnings.warn(f"Large state space size: {joint_state_size}. Computation may be slow.")

    def get_reward_partition(self, reward_value):
        """Get the discretized reward bucket index."""
        index = np.searchsorted(self.s_cutting_points, reward_value, side='right')
        if index == len(self.s_cutting_points):
            index -= 1
        return index - 1
    
    def state_to_index(self, x_states, y, z):
        """
        Convert a joint state to a single index.
        
        Parameters:
        - x_states: array of arm states [x_1, ..., x_n]
        - y: total reward bucket index
        - z: time index
        """
        # Compute joint state index for arms
        x_index = 0
        multiplier = 1
        for i in range(self.num_arms - 1, -1, -1):
            x_index += x_states[i] * multiplier
            multiplier *= self.num_x
        
        # Combine with reward and time indices
        return x_index * self.n_augment * self.num_z + y * self.num_z + z
    
    def index_to_state(self, index):
        """Convert a single index back to joint state (x_states, y, z)."""
        z = index % self.num_z
        temp = index // self.num_z
        y = temp % self.n_augment
        x_index = temp // self.n_augment
        
        # Decode arm states
        x_states = np.zeros(self.num_arms, dtype=int)
        for i in range(self.num_arms - 1, -1, -1):
            x_states[i] = x_index % self.num_x
            x_index //= self.num_x
        
        return x_states, y, z
    
    def compute_immediate_reward(self, x_states, action):
        """
        Compute immediate reward for taking an action in a joint state.
        
        Parameters:
        - x_states: array of arm states
        - action: tuple of arm indices to activate
        """
        total_reward = 0
        for arm_idx in action:
            total_reward += self.rewards[x_states[arm_idx], arm_idx]
        return total_reward
    
    def compute_next_state_distribution(self, x_states, y, z, action):
        """
        Compute the distribution over next states given current state and action.
        
        Returns:
        - next_states: list of (x_states', y', z') tuples
        - probabilities: corresponding probabilities
        """
        # Time always decrements
        next_z = z + 1
        
        # Get current total reward
        current_total_reward = self.all_total_rewards[y]
        
        # Compute immediate reward
        immediate_reward = self.compute_immediate_reward(x_states, action)
        
        # Compute new total reward and its bucket
        new_total_reward = current_total_reward + (self.discount ** z) * immediate_reward
        next_y = self.get_reward_partition(new_total_reward)
        next_y = min(next_y, self.n_augment - 1)  # Ensure within bounds
        
        # Generate all possible next arm states
        action_set = set(action)
        next_states = []
        probabilities = []
        
        # Use iterative approach for all possible transitions
        def generate_transitions(arm_idx, current_x_states, current_prob):
            if arm_idx == self.num_arms:
                next_states.append((current_x_states.copy(), next_y, next_z))
                probabilities.append(current_prob)
                return
            
            x = x_states[arm_idx]
            if arm_idx in action_set:
                # Arm is activated
                for next_x in range(self.num_x):
                    prob = self.transition[x, next_x, 1, arm_idx]
                    if prob > 0:
                        current_x_states[arm_idx] = next_x
                        generate_transitions(arm_idx + 1, current_x_states, current_prob * prob)
                        current_x_states[arm_idx] = x_states[arm_idx]  # Reset
            else:
                # Arm is passive
                for next_x in range(self.num_x):
                    prob = self.transition[x, next_x, 0, arm_idx]
                    if prob > 0:
                        current_x_states[arm_idx] = next_x
                        generate_transitions(arm_idx + 1, current_x_states, current_prob * prob)
                        current_x_states[arm_idx] = x_states[arm_idx]  # Reset
        
        generate_transitions(0, x_states.copy(), 1.0)
        
        return next_states, probabilities
    
    def solve(self):
        """
        Solve for the optimal policy using dynamic programming.
        """
        # Total number of states
        total_states = (self.num_x ** self.num_arms) * self.n_augment * self.num_z
        
        # Initialize value function
        self.V = np.zeros(total_states + self.n_augment, dtype=np.float32)
        
        # Terminal values at z = num_z
        for y in range(self.n_augment):
            terminal_value = self.all_utility_values[y]
            for x_index in range(self.num_x ** self.num_arms):
                state_index = x_index * self.n_augment * self.num_z + y * self.num_z + self.num_z
                self.V[state_index] = terminal_value
        
        # Initialize policy
        self.policy = np.zeros(total_states, dtype=int)
        
        # Backward induction
        for z in range(self.num_z - 1, -1, -1):
            print(f"Solving for time step z = {z}")
            
            for state_idx in range((self.num_x ** self.num_arms) * self.n_augment):
                x_index = state_idx // self.n_augment
                y = state_idx % self.n_augment
                
                # Decode arm states
                x_states = np.zeros(self.num_arms, dtype=int)
                temp_x = x_index
                for i in range(self.num_arms - 1, -1, -1):
                    x_states[i] = temp_x % self.num_x
                    temp_x //= self.num_x
                
                # Evaluate all possible actions
                best_value = -np.inf
                best_action_idx = 0
                
                for action_idx, action in enumerate(self.all_actions):
                    # Compute expected value for this action
                    next_states, probs = self.compute_next_state_distribution(
                        x_states, y, z, action
                    )
                    
                    expected_value = 0
                    for (next_x_states, next_y, next_z), prob in zip(next_states, probs):
                        next_idx = self.state_to_index(next_x_states, next_y, next_z)
                        expected_value += prob * self.V[next_idx]
                    
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action_idx = action_idx
                
                # Store value and policy
                current_idx = self.state_to_index(x_states, y, z)
                self.V[current_idx] = best_value
                self.policy[current_idx] = best_action_idx
    
    def get_action(self, x_states, y, z):
        """
        Get the optimal action for a given state.
        
        Parameters:
        - x_states: array of current arm states
        - y: current total reward bucket index
        - z: current time index
        
        Returns:
        - action_vector: binary vector indicating which arms to activate
        """
        state_idx = self.state_to_index(x_states, y, z)
        action_idx = self.policy[state_idx]
        action_tuple = self.all_actions[action_idx]
        
        # Convert to binary action vector
        action_vector = np.zeros(self.num_arms, dtype=int)
        for arm_idx in action_tuple:
            action_vector[arm_idx] = 1
        
        return action_vector
    
    def get_initial_reward_bucket(self, initial_reward=0):
        """Get the initial reward bucket for starting the process."""
        return self.get_reward_partition(initial_reward)


# Example usage function
def create_optimal_solver(num_states, num_arms, rewards, transition, discount, 
                         u_type, u_order, threshold, m_choices):
    """
    Create and solve for the optimal risk-aware RMAB policy.
    
    Note: This is computationally intensive for large problems!
    """
    solver = OptimalRiskAwareRMAB(
        num_states=num_states,
        num_arms=num_arms,
        rewards=rewards,
        transition=transition,
        discount=discount,
        u_type=u_type,
        u_order=u_order,
        threshold=threshold,
        m_choices=m_choices
    )
    
    # Solve for optimal policy
    solver.solve()
    
    return solver
