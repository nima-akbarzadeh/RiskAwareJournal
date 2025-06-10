import numpy as np
from utils import compute_utility


def generate_random_vector(n, m):
    """
    Generates a random n-length vector with m ones.
    """
    if m > n:
        raise ValueError("Number of ones (m) cannot be greater than vector length (n)")
    
    vector = np.zeros(n, dtype=int)
    ones_indices = np.random.choice(n, m, replace=False)
    vector[ones_indices] = 1
    return vector


def process_random_policy(n_iterations, n_steps, n_states, n_arms, n_choices, threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_iterations, n_steps, n_arms))
    objectives = np.zeros((n_iterations, n_steps, n_arms))
    for k in range(n_iterations):
        states = initial_states.copy()
        for t in range(n_steps):
            actions = generate_random_vector(n_arms, n_choices)
            for a in range(n_arms):
                if t == 0:
                    totalrewards[k, t, a] = rewards[states[a], a]
                else:
                    totalrewards[k, t, a] = totalrewards[k, t-1, a] + rewards[states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
                objectives[k, t, a] = compute_utility(totalrewards[k, t, a], threshold, u_type, u_order)

    return totalrewards, objectives


def generate_greedy_actions(n_arms, n_choices, current_rewards):
    """
    Generates a vector of actions where 1 indicates selecting an arm
    and 0 indicates not selecting it. It selects the n_choices arms
    with the highest current rewards.
    """
    if n_choices > n_arms:
        raise ValueError("Number of choices (n_choices) cannot be greater than number of arms (n_arms)")
    
    # Sort arms by reward in descending order and get their original indices
    sorted_indices = np.argsort(current_rewards)[::-1]
    
    # Select the top n_choices indices
    greedy_indices = sorted_indices[:n_choices]
    
    # Create the actions vector
    actions = np.zeros(n_arms, dtype=int)
    actions[greedy_indices] = 1
    
    return actions


def process_myopic_policy(n_iterations, n_steps, n_states, n_arms, n_choices, threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_iterations, n_steps, n_arms))
    objectives = np.zeros((n_iterations, n_steps, n_arms))
    for k in range(n_iterations):
        states = initial_states.copy()
        for t in range(n_steps):
            actions = generate_greedy_actions(n_arms, n_choices, [rewards[states[a], a] for a in range(n_arms)])
            for a in range(n_arms):
                if t == 0:
                    totalrewards[k, t, a] = rewards[states[a], a]
                else:
                    totalrewards[k, t, a] = totalrewards[k, t-1, a] + rewards[states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
                objectives[k, t, a] = compute_utility(totalrewards[k, t, a], threshold, u_type, u_order)

    return totalrewards, objectives


def process_neutral_whittle(Whittle, n_iterations, n_steps, n_states, n_arms, n_choices, 
                            threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        states = initial_states.copy()
        for t in range(n_steps):
            actions = Whittle.take_action(n_choices, {"x": states, "t": t})
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_neutral_whittle_learning(Whittle, Whittle_learn, n_iterations, n_steps, n_states, n_arms, n_choices, 
                                     threshold, rewards, transitions, initial_states, u_type, u_order):
    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    learn_totalrewards = np.zeros((n_arms, n_iterations))
    learn_objectives = np.zeros((n_arms, n_iterations))
    counts = np.zeros((n_states, n_states, 2, n_arms))
    for k in range(n_iterations):
        states = initial_states.copy()
        learn_states = initial_states.copy()
        for t in range(n_steps):
            actions = Whittle.take_action(n_choices, {"x": states, "t": t})
            learn_actions = Whittle_learn.take_action(n_choices, {"x": learn_states, "t": t})
            _learn_states = np.copy(learn_states)
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                learn_totalrewards[a, k] += rewards[learn_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
                learn_states[a] = np.random.choice(n_states, p=transitions[learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)
            learn_objectives[a, k] = compute_utility(learn_totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts


def process_riskaware_whittle(raWhittle, n_iterations, n_steps, n_states, n_arms, n_choices, 
                              threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        lifted = np.zeros(n_arms, dtype=np.int32)
        states = initial_states.copy()
        for t in range(n_steps):
            actions = raWhittle.take_action(n_choices, {"l": lifted, "x": states, "t": t})
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                lifted[a] = max(0, min(raWhittle.n_augment[a]-1, lifted[a] + states[a]))
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_riskaware_whittle_learning(raWhittle, raWhittle_learn, n_iterations, n_steps, n_states, n_arms, 
                                       n_choices, threshold, rewards, transitions, initial_states, u_type, u_order):
    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    learn_totalrewards = np.zeros((n_arms, n_iterations))
    learn_objectives = np.zeros((n_arms, n_iterations))
    counts = np.zeros((n_states, n_states, 2, n_arms))
    for k in range(n_iterations):
        lifted = np.zeros(n_arms, dtype=np.int32)
        states = initial_states.copy()
        learn_lifted = np.zeros(n_arms, dtype=np.int32)
        learn_states = initial_states.copy()
        for t in range(n_steps):
            actions = raWhittle.take_action(n_choices, {"l": lifted, "x": states, "t": t})
            learn_actions = raWhittle_learn.take_action(n_choices, {"l": learn_lifted, "x": learn_states, "t": t})
            _learn_states = np.copy(learn_states)
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], a]
                lifted[a] = max(0, min(raWhittle.n_augment[a]-1, lifted[a] + states[a]))
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
                learn_totalrewards[a, k] += rewards[learn_states[a], a]
                learn_lifted[a] = max(0, min(raWhittle_learn.n_augment[a]-1, learn_lifted[a] + learn_states[a]))
                learn_states[a] = np.random.choice(n_states, p=transitions[learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)
            learn_objectives[a, k] = compute_utility(learn_totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts


def process_ns_neutral_whittle(Whittle, n_iterations, n_steps, n_states, n_arms, n_choices, 
                               threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        states = initial_states.copy()
        for t in range(n_steps):
            actions = Whittle.take_action(n_choices, {"x": states, "t": t})
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], t, a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_ns_riskaware_whittle(raWhittle, n_iterations, n_steps, n_states, n_arms, n_choices, 
                                 threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        lifted = np.zeros(n_arms, dtype=np.int32)
        states = initial_states.copy()
        for t in range(n_steps):
            actions = raWhittle.take_action(n_choices, {"l": lifted, "x": states, "t": t})
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], t, a]
                lifted[a] = raWhittle.get_reward_partition(totalrewards[a, k])
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_ns_riskaware_whittle_learning(raWhittle, raWhittle_learn, n_iterations, n_steps, n_states, n_arms, n_choices, 
                                          threshold, rewards, transitions, initial_states, u_type, u_order):
    
    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    learn_totalrewards = np.zeros((n_arms, n_iterations))
    learn_objectives = np.zeros((n_arms, n_iterations))
    counts = np.zeros((n_states, n_states, 2, n_arms))
    for k in range(n_iterations):
        lifted = np.zeros(n_arms, dtype=np.int32)
        states = initial_states.copy()
        learn_lifted = np.zeros(n_arms, dtype=np.int32)
        learn_states = initial_states.copy()
        for t in range(n_steps):
            actions = raWhittle.take_action(n_choices, {"l": lifted, "x": states, "t": t})
            learn_actions = raWhittle_learn.take_action(n_choices, {"l": learn_lifted, "x": learn_states, "t": t})
            _learn_states = np.copy(learn_states)
            for a in range(n_arms):
                totalrewards[a, k] += rewards[states[a], t, a]
                lifted[a] = raWhittle.get_reward_partition(totalrewards[a, k])
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
                learn_totalrewards[a, k] += rewards[learn_states[a], t, a]
                learn_lifted[a] = raWhittle.get_reward_partition(learn_totalrewards[a, k])
                learn_states[a] = np.random.choice(n_states, p=transitions[learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)
            learn_objectives[a, k] = compute_utility(learn_totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts


def process_inf_neutral_whittle(Whittle, n_iterations, discount, n_steps, n_states, n_arms, n_choices, 
                                threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        states = initial_states.copy()
        for t in range(n_steps):
            discount_val = discount ** t
            actions = Whittle.take_action(n_choices, {"x": states})
            for a in range(n_arms):
                totalrewards[a, k] += discount_val * rewards[states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_inf_riskaware_whittle(raWhittle, Whittle, n_discount, n_iterations, discount, n_steps, n_states, n_arms, n_choices, 
                                  threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        lifted = np.zeros(n_arms, dtype=np.int32)
        states = initial_states.copy()
        for t in range(n_steps):
            discount_val = discount ** t
            if t < n_discount:
                
                actions = raWhittle.take_action(n_choices, {"l": lifted, "x": states, "t": t})
            else:
                actions = Whittle.take_action(n_choices, {"x": states})
            for a in range(n_arms):
                totalrewards[a, k] += discount_val * rewards[states[a], a]
                if t < n_discount:
                    lifted[a] = raWhittle.get_reward_partition(totalrewards[a, k])
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_inf_riskaware_whittle_learning(raWhittle, raWhittle_learn, discount, n_steps, n_states, n_arms, n_choices, 
                                           threshold, rewards, transitions, initial_states, u_type, u_order):
    
    totalrewards = np.zeros(n_arms)
    objectives = np.zeros(n_arms)
    lifted = np.zeros(n_arms, dtype=np.int32)
    states = initial_states.copy()
    learn_totalrewards = np.zeros(n_arms)
    learn_objectives = np.zeros(n_arms)
    learn_lifted = np.zeros(n_arms, dtype=np.int32)
    learn_states = initial_states.copy()
    counts = np.zeros((n_states, n_states, 2, n_arms))
    for t in range(n_steps):
        discount_val = discount ** t
        actions = raWhittle.take_action(n_choices, {"l": lifted, "x": states, "t": t})
        learn_actions = raWhittle_learn.take_action(n_choices, {"l": learn_lifted, "x": learn_states, "t": t})
        _learn_states = np.copy(learn_states)
        for a in range(n_arms):
            totalrewards[a] += discount_val * rewards[states[a], a]
            lifted[a] = raWhittle.get_reward_partition(totalrewards[a])
            states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
            learn_totalrewards[a] += discount_val * rewards[learn_states[a], a]
            learn_lifted[a] = raWhittle_learn.get_reward_partition(learn_totalrewards[a])
            learn_states[a] = np.random.choice(n_states, p=transitions[learn_states[a], :, learn_actions[a], a])
            counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
    for a in range(n_arms):
        objectives[a] = compute_utility(totalrewards[a], threshold, u_type, u_order)
        learn_objectives[a] = compute_utility(learn_totalrewards[a], threshold, u_type, u_order)

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts
