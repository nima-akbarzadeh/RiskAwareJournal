import numpy
from Markov import rewards, get_transitions

def check_conditions(P, r):
    """
    Check if the transition matrix and rewards satisfy the given assumptions and theorem.

    Parameters:
    - P: 3D NumPy array of shape (len(X), len(X), len(A)) representing the transition probabilities.
    - r: 2D NumPy array of shape (len(X), len(A)) representing the reward function.

    Returns:
    - dict: A dictionary indicating which conditions are satisfied.
    """
    num_states, _, num_actions = P.shape

    results = {
        "r_nondecreasing_in_x": True,
        "q_nondecreasing_in_x": True,
        "r_super_additive": True,
        "q_super_additive": True,
        "q_nondecreasing_in_a": True,
        "r_independent_of_a": True,
    }

    # Check if r(x, a) is nondecreasing in x for all a
    for a in range(num_actions):
        if not numpy.all(numpy.diff(r[:, a]) >= 0):
            results["r_nondecreasing_in_x"] = False

    # Check if q^i(x'|x, a) is nondecreasing in x for all x', a
    for x_prime in range(num_states):
        for a in range(num_actions):
            q_x = numpy.cumsum(P[:, x_prime, a][::-1])[::-1]  # Calculate q(x|x, a) = sum_{z >= k} P(z|x, a)
            if not numpy.all(numpy.diff(q_x) >= 0):
                results["q_nondecreasing_in_x"] = False

    # Check if r(x, a) is super-additive
    for x1 in range(num_states):
        for x2 in range(num_states):
            for a in range(num_actions):
                if r[x1, a] + r[x2, a] > r[min(x1 + x2, num_states - 1), a]:
                    results["r_super_additive"] = False

    # Check if q^i(x'|x, a) is super-additive
    for x1 in range(num_states):
        for x2 in range(num_states):
            for x_prime in range(num_states):
                for a in range(num_actions):
                    q1 = numpy.cumsum(P[x1, :, a][::-1])[::-1][x_prime]
                    q2 = numpy.cumsum(P[x2, :, a][::-1])[::-1][x_prime]
                    combined_q = numpy.cumsum(P[min(x1 + x2, num_states - 1), :, a][::-1])[::-1][x_prime]
                    if q1 + q2 > combined_q:
                        results["q_super_additive"] = False

    # Check if q^i(x'|x, a) is nondecreasing in a for all x, x'
    for x in range(num_states):
        for x_prime in range(num_states):
            for a in range(num_actions - 1):
                if P[x, x_prime, a] > P[x, x_prime, a + 1]:
                    results["q_nondecreasing_in_a"] = False

    # Check if r(x, a) = r(x) for all x, a
    for a in range(num_actions - 1):
        if not numpy.all(r[:, a] == r[:, a + 1]):
            results["r_independent_of_a"] = False

    return results

# Example Usage
# Define a transition matrix P and a reward function r
T = 5 # Time horizon
N = 1 # Number of arms
X = 3  # Number of states
A = 2  # Number of actions

r = rewards(T, A, X)
pr_ss_0 = numpy.round(numpy.linspace(0.657, 0.762, N), 3)
numpy.random.shuffle(pr_ss_0)
pr_sp_0 = numpy.round(numpy.linspace(0.201, 0.287, N), 3)
numpy.random.shuffle(pr_sp_0)
pr_pp_0 = numpy.round(numpy.linspace(0.882, 0.922, N), 3)
numpy.random.shuffle(pr_pp_0)
pr_ss_1 = numpy.round(numpy.linspace(0.806, 0.869, N), 3)
numpy.random.shuffle(pr_ss_1)
pr_sp_1 = numpy.round(numpy.linspace(0.115, 0.171, N), 3)
numpy.random.shuffle(pr_sp_1)
pr_pp_1 = numpy.round(numpy.linspace(0.879, 0.921, N), 3)
numpy.random.shuffle(pr_pp_1)
prob_remain = [pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1]
Pr = get_transitions(N, X, prob_remain, 'clinical')
P = Pr[:, :, :, 0]

# Check conditions
conditions = check_conditions(P, r)
print("Condition Check Results:")
for condition, satisfied in conditions.items():
    print(f"{condition}: {satisfied}")
