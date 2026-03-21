import matplotlib.pyplot as plt
import numpy as np

def annotated_heatmap(data, title):
    ylabels = ["states"]
    xlabels_idx = range(len(data[0]))
    xlabels= ["s"+str(x) for x in xlabels_idx]

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(xlabels)), labels=xlabels, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(ylabels)), rotation=90, labels=ylabels)

    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            text = ax.text(j, i, round(data[i][j], 1),
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

# Build transition matrix and reward vector under fixed policy
def analytical_policy_evaluation(pi, model, gamma):
    n_states = len(pi)
    n_actions = len(pi[0])
    P = np.zeros((n_states, n_states))
    R = np.zeros(n_states)

    for s in range(n_states):
        for a in range(n_actions):
            for prob, s_prime, reward, done in model[s][a]:
                P[s, s_prime] = pi[s][a]*prob*(not done)
                R[s] += pi[s][a]*reward*prob

    I = np.eye(n_states)
    V = np.linalg.inv(I - gamma * P) @ R

    np.set_printoptions(precision=1, suppress=True)
    return V
