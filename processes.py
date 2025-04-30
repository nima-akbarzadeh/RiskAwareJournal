import numpy as np


def compute_utility(total_reward, threshold, u_type, u_order):
    if u_type == 1:
        if total_reward - threshold >= 0:
            return 1
        else:
            return 0
    elif u_type == 2:
        return 1 - threshold**(- 1/u_order) * (np.maximum(0, threshold - total_reward))**(1/u_order)
    else:
        return (1 + np.exp(-u_order * (1 - threshold))) / (1 + np.exp(-u_order * (total_reward - threshold)))


def process_neutral_whittle(Whittle, n_iterations, n_steps, n_states, n_arms, n_choices, 
                            threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        states = initial_states.copy()
        for t in range(n_steps):
            actions = Whittle.take_action(n_choices, states, t)
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
            actions = Whittle.take_action(n_choices, states, t)
            learn_actions = Whittle_learn.take_action(n_choices, learn_states, t)
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
            actions = raWhittle.take_action(n_choices, lifted, states, t)
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
            actions = raWhittle.take_action(n_choices, lifted, states, t)
            learn_actions = raWhittle_learn.take_action(n_choices, learn_lifted, learn_states, t)
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
            actions = Whittle.take_action(n_choices, states, t)
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
            actions = raWhittle.take_action(n_choices, lifted, states, t)
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
            actions = raWhittle.take_action(n_choices, lifted, states, t)
            learn_actions = raWhittle_learn.take_action(n_choices, learn_lifted, learn_states, t)
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
            actions = Whittle.take_action(n_choices, states)
            for a in range(n_arms):
                totalrewards[a, k] += discount_val * rewards[states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_inf_riskaware_whittle(raWhittle, n_iterations, discount, n_steps, n_states, n_arms, n_choices, 
                                  threshold, rewards, transitions, initial_states, u_type, u_order):

    totalrewards = np.zeros((n_arms, n_iterations))
    objectives = np.zeros((n_arms, n_iterations))
    for k in range(n_iterations):
        lifted = np.zeros(n_arms, dtype=np.int32)
        states = initial_states.copy()
        for t in range(n_steps):
            discount_val = discount ** t
            discount_idx = raWhittle.get_discnt_partition(discount_val)
            actions = raWhittle.take_action(n_choices, lifted, discount_idx, states)
            for a in range(n_arms):
                totalrewards[a, k] += discount_val * rewards[states[a], a]
                lifted[a] = raWhittle.get_reward_partition(totalrewards[a, k])
                states[a] = np.random.choice(n_states, p=transitions[states[a], :, actions[a], a])
        for a in range(n_arms):
            objectives[a, k] = compute_utility(totalrewards[a, k], threshold, u_type, u_order)

    return totalrewards, objectives


def process_inf_riskaware_whittle_learning(raWhittle, raWhittle_learn, discount, n_steps, n_states, n_arms, n_choices, 
                                           threshold, rewards, transitions, initial_states, u_type, u_order):
    
    totalrewards = np.zeros(n_arms)
    objectives = np.zeros(n_arms)
    learn_totalrewards = np.zeros(n_arms)
    learn_objectives = np.zeros(n_arms)
    counts = np.zeros((n_states, n_states, 2, n_arms))
    lifted = np.zeros(n_arms, dtype=np.int32)
    states = initial_states.copy()
    learn_lifted = np.zeros(n_arms, dtype=np.int32)
    learn_states = initial_states.copy()
    for t in range(n_steps):
        discount_val = discount ** t
        discount_idx = raWhittle.get_discnt_partition(discount_val)
        actions = raWhittle.take_action(n_choices, lifted, discount_idx, states)
        learn_actions = raWhittle_learn.take_action(n_choices, learn_lifted, discount_idx, learn_states)
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
