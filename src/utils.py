import matplotlib.pyplot as plt

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
