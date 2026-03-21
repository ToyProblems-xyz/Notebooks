import numpy as np
from utils import analytical_policy_evaluation

def rollout_episode(env, pi):
    done = False
    possible_actions = np.linspace(0, env.nA-1, env.nA)
    s = np.random.randint(0,env.nS)
    trajectory = []    
    while not done:
        a = np.random.choice(possible_actions, p=pi[s])
        [p, s_, r, done] = env.step(s, a)[0]
        transition = [s, int(a), r]
        trajectory.append(transition)
        s = s_
    return trajectory

def monte_carlo_prediction(env, pi, gamma: float = 1.0, num_episodes: int = 10):
    n_states = len(pi)
    n_actions = len(pi[0])
    
    # Create unique dictionaries for every state and action using list comprehension because dict is mutable
    returns = [[{"avg_return": 0.0, "freq": 0} for _ in range(n_actions)] for _ in range(n_states)]

    # Similarly for the q-table to be safe using list comprehension because list is mutable
    q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    v = [0.0]*n_states # this can be defined using "*" instead of list comprehension because int is immutable

    err = []
    v_analytical = analytical_policy_evaluation(pi, env.model, gamma=gamma)

    for e in range(num_episodes):
        trajectory = rollout_episode(env, pi)
        G = 0
        for s, a, r in reversed(trajectory):
            G = G*gamma + r
            new_freq = returns[s][a]["freq"] + 1
            returns[s][a]["avg_return"] = (returns[s][a]["avg_return"]*returns[s][a]["freq"] + G)/new_freq
            returns[s][a]["freq"] = new_freq
            q[s][a] = returns[s][a]["avg_return"]
            v[s] = max(q[s])
        v[-1] = 0.0
        err.append(sum((v_analytical - v)**2)/n_states)
    return v, err