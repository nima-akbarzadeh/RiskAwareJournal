import numpy as np

num_states = 4
time_horizon = 10
num_arms = 1
discount = 0.9
vals = np.ones((num_states, time_horizon, num_arms))
for t in range(time_horizon):
    per_step_rewards = np.round((1 - discount) * (discount**t) * np.linspace(0, 1, num=num_states)  / (1 - discount ** time_horizon), 3)
    for x, per_step_reward in enumerate(per_step_rewards):
        vals[x, t, :] = per_step_reward * np.ones(num_arms)
print(vals)
