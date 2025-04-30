### Risk-Neutral & Risk-Aware Whittle Index
import numpy as np
from itertools import product
from processes import compute_utility


def possible_reward_sums(rewards, num_steps):
    reward_combinations = product(rewards, repeat=num_steps)
    sums = set()
    for combination in reward_combinations:
        sums.add(np.round(sum(combination), 3))
    return sorted(sums)


class Whittle:

    def __init__(self, num_states: int, num_arms: int, reward, transition, horizon):
        self.num_x = num_states
        self.num_a = num_arms
        self.reward = reward
        self.transition = transition
        self.horizon = horizon
        self.digits = 3
        self.whittle_indices = []

    def get_indices(self, index_range, n_trials):
        l_steps = index_range / n_trials
        self.binary_search(0, index_range, l_steps)

    def is_equal_mat(self, mat1, mat2, tol=1e-6):
        return np.all(np.abs(mat1 - mat2) < tol)

    def indexability_check(self, arm_indices, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            if np.any((ref_pol[:, t] == 0) & (nxt_pol[:, t] == 1)):
                print("Neutral - Not indexable!")
                return False, np.zeros((self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol[:, t] == 1) & (nxt_pol[:, t] == 0))
                for e in elements:
                    arm_indices[e, t] = penalty
        return True, arm_indices

    def binary_search(self, lower_bound, upper_bound, l_steps):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward(arm, penalty_ref)
            ubp_pol, _, _ = self.backward(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = 0.5 * (lb_temp + ub_temp)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = 0.5 * (lb_temp + ub_temp)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.backward(arm, penalty_ref)
                flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                if flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.whittle_indices.append(arm_indices)

    def backward(self, arm, penalty):
        # Value function initialization
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)

        # State-action value function
        Q = np.zeros((self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        for t in range(self.horizon - 1, -1, -1):
            for x in range(self.num_x):
                # Calculate Q-values for both actions
                Q[x, t, 0] = self.reward[x, arm] + np.dot(V[:, t + 1], self.transition[x, :, 0, arm])
                Q[x, t, 1] = self.reward[x, arm] - penalty / self.horizon + np.dot(V[:, t + 1], self.transition[x, :, 1, arm])

                # Optimal action and value
                pi[x, t] = np.argmax(Q[x, t, :])
                V[x, t] = np.max(Q[x, t, :])

        return pi, V, Q

    def take_action(self, n_choices, current_x, current_t):

        current_indices = np.zeros(self.num_a)
        count_positive = 0
        for arm in range(self.num_a):
            w_idx = self.whittle_indices[arm][current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_choices = np.minimum(n_choices, count_positive)

        max_index = np.max(current_indices)
        candidates = np.where(current_indices == max_index)[0]
        chosen = np.random.choice(candidates, size=min(n_choices, len(candidates)), replace=False)
        action_vector = np.zeros_like(current_indices, dtype=int)
        action_vector[chosen] = 1

        return action_vector


class RiskAwareWhittle:
    
    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, threshold):
        self.num_x = num_states
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * self.num_a
        self.all_rews = []
        self.all_utility_values = []

        for a in range(self.num_a):

            all_immediate_rew = self.rewards[:, a]
            arm_n_realize = []
            all_total_rewards = []
            for t in range(self.horizon):
                all_total_rewards_by_t = possible_reward_sums(all_immediate_rew.flatten(), t + 1)
                arm_n_realize.append(len(all_total_rewards_by_t))
                if t == self.horizon - 1:
                    all_total_rewards = all_total_rewards_by_t

            self.n_augment[a] = len(all_total_rewards)
            self.all_rews.append(all_total_rewards)
            self.n_realize.append(arm_n_realize)

            arm_utilities = []
            for total_reward in all_total_rewards:
                arm_utilities.append(compute_utility(total_reward, threshold, u_type, u_order))
            self.all_utility_values.append(arm_utilities)

        self.whittle_indices = []

    def get_indices(self, index_range, n_trials):
        l_steps = index_range / n_trials
        self.binary_search(0, index_range, l_steps)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                print("RA - Not indexable!")
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward_discreteliftedstate(arm, penalty_ref)
            ubp_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol, self.n_realize[arm]):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward_discreteliftedstate(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol, self.n_realize[arm]):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.whittle_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_utility_values[arm][l] * np.ones(self.num_x)

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(self.n_realize[arm][t]):

                    nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))
                    
                    Q[l, x, t, 0] = np.dot(V[nxt_l, :, t + 1], self.transition[x, :, 0, arm])
                    Q[l, x, t, 1] = - penalty / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, 1, arm])

                    # Get the value function and the policy
                    pi[l, x, t] = np.argmax(Q[l, x, t, :])
                    V[l, x, t] = np.max(Q[l, x, t, :])

            t = t - 1
        
        return pi, V, Q

    def take_action(self, n_choices, current_l, current_x, current_t):

        current_indices = np.zeros(self.num_a)
        count_positive = 0
        for arm in range(self.num_a):
            w_idx = self.whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_choices = np.minimum(n_choices, count_positive)

        max_index = np.max(current_indices)
        candidates = np.where(current_indices == max_index)[0]
        chosen = np.random.choice(candidates, size=min(n_choices, len(candidates)), replace=False)
        action_vector = np.zeros_like(current_indices, dtype=int)
        action_vector[chosen] = 1

        return action_vector


class WhittleNS:

    def __init__(self, num_states: int, num_arms: int, reward, transition, horizon):
        self.num_x = num_states
        self.num_a = num_arms
        self.reward = reward
        self.transition = transition
        self.horizon = horizon
        self.digits = 3
        self.whittle_indices = []

    def get_indices(self, index_range, n_trials):
        l_steps = index_range / n_trials
        self.binary_search(0, index_range, l_steps)

    def is_equal_mat(self, mat1, mat2, tol=1e-6):
        return np.all(np.abs(mat1 - mat2) < tol)

    def indexability_check(self, arm_indices, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            if np.any((ref_pol[:, t] == 0) & (nxt_pol[:, t] == 1)):
                print("Neutral - Not indexable!")
                return False, np.zeros((self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol[:, t] == 1) & (nxt_pol[:, t] == 0))
                for e in elements:
                    arm_indices[e, t] = penalty
        return True, arm_indices

    def binary_search(self, lower_bound, upper_bound, l_steps):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward(arm, penalty_ref)
            ubp_pol, _, _ = self.backward(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = 0.5 * (lb_temp + ub_temp)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = 0.5 * (lb_temp + ub_temp)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.backward(arm, penalty_ref)
                flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                if flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.whittle_indices.append(arm_indices)

    def backward(self, arm, penalty):
        # Value function initialization
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)

        # State-action value function
        Q = np.zeros((self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        for t in range(self.horizon - 1, -1, -1):
            for x in range(self.num_x):

                # Calculate Q-values for both actions
                Q[x, t, 0] = self.reward[x, t, arm] + np.dot(V[:, t + 1], self.transition[x, :, 0, arm])
                Q[x, t, 1] = self.reward[x, t, arm] - penalty / self.horizon + np.dot(V[:, t + 1], self.transition[x, :, 1, arm])

                # Optimal action and value
                pi[x, t] = np.argmax(Q[x, t, :])
                V[x, t] = np.max(Q[x, t, :])

        return pi, V, Q

    def take_action(self, n_choices, current_x, current_t):

        current_indices = np.zeros(self.num_a)
        count_positive = 0
        for arm in range(self.num_a):
            w_idx = self.whittle_indices[arm][current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_choices = np.minimum(n_choices, count_positive)

        max_index = np.max(current_indices)
        candidates = np.where(current_indices == max_index)[0]
        chosen = np.random.choice(candidates, size=min(n_choices, len(candidates)), replace=False)
        action_vector = np.zeros_like(current_indices, dtype=int)
        action_vector[chosen] = 1

        return action_vector


class RiskAwareWhittleNS:
    
    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, threshold):
        self.num_x = num_states[0]
        self.num_s = num_states[1]
        self.cutting_points = np.round(np.linspace(0, horizon, self.num_s+1), 2)
        self.all_total_rewards = np.round([np.median(self.cutting_points[i:i + 2]) for i in range(len(self.cutting_points) - 1)], 2)
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * self.num_a
        self.all_rews = []
        self.all_utility_values = []

        for a in range(num_arms):
            self.n_augment[a] = len(self.all_total_rewards)
            self.n_realize.append([self.num_s] * self.horizon)
            self.all_rews.append(self.all_total_rewards)

            arm_utilities = []
            for total_reward in self.all_total_rewards:
                arm_utilities.append(compute_utility(total_reward, threshold, u_type, u_order))
            self.all_utility_values.append(arm_utilities)

        self.whittle_indices = []

    def get_reward_partition(self, reward_value):
        index = np.searchsorted(self.cutting_points, reward_value, side='right')
        if index == len(self.cutting_points):
            index -= 1

        return index - 1

    def get_indices(self, index_range, n_trials):
        l_steps = index_range / n_trials
        self.binary_search(0, index_range, l_steps)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                print("RA - Not indexable!")
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward_discreteliftedstate(arm, penalty_ref)
            ubp_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol, self.n_realize[arm]):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward_discreteliftedstate(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol, self.n_realize[arm]):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.whittle_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_utility_values[arm][l] * np.ones(self.num_x)

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(len(self.all_total_rewards)):

                    nxt_l = self.get_reward_partition(self.all_total_rewards[l] + self.rewards[x, t, arm])
                    
                    Q[l, x, t, 0] = np.dot(V[nxt_l, :, t + 1], self.transition[x, :, 0, arm])
                    Q[l, x, t, 1] = - penalty / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, 1, arm])

                    # Get the value function and the policy
                    pi[l, x, t] = np.argmax(Q[l, x, t, :])
                    V[l, x, t] = np.max(Q[l, x, t, :])

            t = t - 1
        
        return pi, V, Q

    def take_action(self, n_choices, current_l, current_x, current_t):

        current_indices = np.zeros(self.num_a)
        count_positive = 0
        for arm in range(self.num_a):
            w_idx = self.whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_choices = np.minimum(n_choices, count_positive)

        max_index = np.max(current_indices)
        candidates = np.where(current_indices == max_index)[0]
        chosen = np.random.choice(candidates, size=min(n_choices, len(candidates)), replace=False)
        action_vector = np.zeros_like(current_indices, dtype=int)
        action_vector[chosen] = 1

        return action_vector


class WhittleInf:

    def __init__(self, num_states: int, num_arms: int, reward, transition, discount):
        
        self.discount = discount
        self.num_x = num_states
        self.num_a = num_arms
        self.reward = reward
        self.transition = transition
        self.digits = 3
        self.whittle_indices = []

    def get_indices(self, index_range, n_trials):
        l_steps = index_range / n_trials
        self.binary_search(0, index_range, l_steps)

    def is_equal_mat(self, mat1, mat2, tol=1e-6):
        return np.all(np.abs(mat1 - mat2) < tol)

    def indexability_check(self, arm_indices, nxt_pol, ref_pol, penalty):
        if np.any((ref_pol == 0) & (nxt_pol == 1)):
            print("Neutral - Not indexable!")
            return False, np.zeros(self.num_x)
        else:
            elements = np.argwhere((ref_pol == 1) & (nxt_pol == 0))
            for e in elements:
                arm_indices[e] = penalty
        return True, arm_indices

    def binary_search(self, lower_bound, upper_bound, l_steps):
        for arm in range(self.num_a):
            arm_indices = np.zeros(self.num_x)
            penalty_ref = lower_bound
            ref_pol, _, _ = self.bellman(arm, penalty_ref)
            ubp_pol, _, _ = self.bellman(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = 0.5 * (lb_temp + ub_temp)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.bellman(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = 0.5 * (lb_temp + ub_temp)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.bellman(arm, penalty_ref)
                flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                if flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.whittle_indices.append(arm_indices)

    def bellman(self, arm, penalty):
        # Value function initialization
        V = np.zeros(self.num_x, dtype=np.float32)

        # State-action value function
        Q = np.zeros((self.num_x, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros(self.num_x, dtype=np.int32)

        # Value iteration
        diff = np.inf
        iteration = 0
        while diff > 1e-4 and iteration < 1000:
            V_prev = np.copy(V)
            for x in range(self.num_x):
                # Calculate Q-values for both actions
                for a in range(2):
                    Q[x, a] = self.reward[x, arm] - penalty * a + self.discount * np.dot(V, self.transition[x, :, a, arm])

                # Optimal action and value
                pi[x] = np.argmax(Q[x, :])
                V[x] = np.max(Q[x, :])
            diff = np.max(np.abs(V - V_prev))
            iteration += 1

        return pi, V, Q

    def take_action(self, n_choices, current_x):

        current_indices = np.zeros(self.num_a)
        count_positive = 0
        for arm in range(self.num_a):
            w_idx = self.whittle_indices[arm][current_x[arm]]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_choices = np.minimum(n_choices, count_positive)

        max_index = np.max(current_indices)
        candidates = np.where(current_indices == max_index)[0]
        chosen = np.random.choice(candidates, size=min(n_choices, len(candidates)), replace=False)
        action_vector = np.zeros_like(current_indices, dtype=int)
        action_vector[chosen] = 1

        return action_vector


class RiskAwareWhittleInf:
    
    def __init__(self, num_states: int, num_arms: int, rewards, transition, discount, u_type, u_order, threshold):
        
        self.discount = discount
        self.num_x = num_states[0]
        self.num_s = num_states[1]
        self.num_z = num_states[2]
        self.s_cutting_points = np.linspace(0, 1, self.num_s+1)
        self.z_cutting_points = np.linspace(0, 1, self.num_z+1)
        self.all_total_rewards = [np.round(np.median(self.s_cutting_points[i:i + 2]), 3) for i in range(len(self.s_cutting_points) - 1)]
        self.all_total_discnts = [np.round(np.median(self.z_cutting_points[i:i + 2]), 3) for i in range(len(self.z_cutting_points) - 1)]
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.u_type = u_type
        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * self.num_a
        self.all_rews = []
        self.all_utility_values = []

        for a in range(self.num_a):
            self.n_augment[a] = len(self.all_total_rewards)

            arm_utilities = []
            for total_reward in self.all_total_rewards:
                arm_utilities.append(compute_utility(total_reward, threshold, u_type, u_order))
            self.all_utility_values.append(arm_utilities)

        self.whittle_indices = []

    def get_reward_partition(self, reward_value):
        index = np.searchsorted(self.s_cutting_points, reward_value, side='right')
        if index == len(self.s_cutting_points):
            index -= 1

        return index - 1
    
    def get_discnt_partition(self, discnt_value):
        index = np.searchsorted(self.z_cutting_points, discnt_value, side='right')
        if index == len(self.z_cutting_points):
            index -= 1

        return index - 1

    def get_indices(self, index_range, n_trials):
        l_steps = index_range / n_trials
        self.binary_search(0, index_range, l_steps)

    def is_equal_mat(self, mat1, mat2):
        if np.array_equal(mat1, mat2):
            return True
        return False

    def indexability_check(self, arm, arm_indices, nxt_pol, ref_pol, nxt_Q, ref_Q, penalty):
        if np.any((ref_pol == 0) & (nxt_pol == 1)):
            print("="*50)
            print("RA - Not indexable!")
            elements = np.argwhere((ref_pol == 0) & (nxt_pol == 1))
            for e in elements:
                print(f"e = {e}")
                print(f"ref_Q[e[0], e[1], e[2]] = {ref_Q[e[0], e[1], e[2]]}")
                print(f"nxt_Q[e[0], e[1], e[2]] = {nxt_Q[e[0], e[1], e[2]]}")
            return False, np.zeros((self.n_augment[arm], self.num_z, self.num_x))
        else:
            elements = np.argwhere((ref_pol == 1) & (nxt_pol == 0))
            for e in elements:
                arm_indices[e[0], e[1], e[2]] = penalty
        return True, arm_indices

    def binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_z, self.num_x))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.bellman(arm, penalty_ref)
            ubp_pol, _, _ = self.bellman(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.bellman(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, nxt_Q = self.bellman(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, nxt_pol, ref_pol, nxt_Q, ref_Q, penalty)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                    ref_Q = np.copy(nxt_Q)
                else:
                    break
            self.whittle_indices.append(arm_indices)

    def bellman(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_z, self.num_x), dtype=np.float32)
        for y in range(self.n_augment[arm]):
            V[y, :, :] = self.all_utility_values[arm][y] * np.ones([self.num_z, self.num_x])

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_z, self.num_x, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_z, self.num_x), dtype=np.int32)

        # Value iteration
        diff = np.inf
        iteration = 0
        while diff > 1e-4 and iteration < 1000:
            V_prev = np.copy(V)

            # Loop over all dimensions of the state space
            for x in range(self.num_x):
                for z in range(self.num_z):
                    for y in range(self.n_augment[arm]):
                        nxt_y = self.get_reward_partition(self.all_total_rewards[y] + self.all_total_discnts[z] * self.rewards[x, arm])
                        nxt_z = self.get_discnt_partition(self.all_total_discnts[z] * self.discount)
                        for a in range(2):
                            Q[y, z, x, a] = - penalty * self.all_total_discnts[z] * a + np.dot(V[nxt_y, nxt_z, :], self.transition[x, :, a, arm])

                        # Get the value function and the policy
                        pi[y, z, x] = np.argmax(Q[y, z, x, :])
                        V[y, z, x] = np.max(Q[y, z, x, :])

            diff = np.max(np.abs(V - V_prev))
            iteration += 1

        return pi, V, Q

    def take_action(self, n_choices, current_l, current_z, current_x):

        current_indices = np.zeros(self.num_a)
        count_positive = 0
        for arm in range(self.num_a):
            w_idx = self.whittle_indices[arm][current_l[arm], current_z, current_x[arm]]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_choices = np.minimum(n_choices, count_positive)

        max_index = np.max(current_indices)
        candidates = np.where(current_indices == max_index)[0]
        chosen = np.random.choice(candidates, size=min(n_choices, len(candidates)), replace=False)
        action_vector = np.zeros_like(current_indices, dtype=int)
        action_vector[chosen] = 1

        return action_vector

