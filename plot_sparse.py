import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("sparse-compressed-distributions.csv")

baseline_labels = {
    # 2: 'Naive',
    # 3: 'Butterfly',
    # 4: 'Tree',
    # 5: 'Ring',
    11: 'Naive',
    12: 'Delta',
    13: 'Bitmap',
}

distribution_labels = {
    1: 'Uniform',
    2: 'Exponential',
}

# remove the rows from df that don't have a Baseline value in the baseline_labels
df = df[df['Baseline'].isin(baseline_labels.keys())]


# densities = sorted(df['Density'].unique())
# baselines = sorted(df['Baseline'].unique())

baselines = [11, 13, 12]

densities = [0.01]
distributions = [1, 2]

colors = ['blue', 'orange', 'green', 'red', 'purple']
markers = ['o', 's', 'd', '^', '*']

baseline_style = {
    # 2: (colors[0], markers[0]),
    # 3: (colors[1], markers[1]),
    # 4: (colors[2], markers[2]),
    # 5: (colors[3], markers[3]),
    11: (colors[0], markers[0]),
    12: (colors[1], markers[1]),
    13: (colors[2], markers[2]),
}

os.makedirs("plots", exist_ok=True)

# for density in densities:
#     plt.figure(figsize=(8, 6))
    
#     for baseline in baselines:
#         subset = df[(df['Density'] == density) & (df['Baseline'] == baseline)]
#         subset = subset.sort_values('TasksPerNode')

#         color, marker = baseline_style.get(baseline, ('black', 'o'))
#         plt.plot(subset['TasksPerNode'], subset['Time'],
#                  marker=marker, color=color, label=baseline_labels.get(baseline, f"Baseline {baseline}"))
    
#     plt.title(f"Sparsity Complexity (Density = {density})")
#     plt.xlabel("Number of Processes")
#     plt.ylabel("Simulation Time (seconds)")
#     plt.xscale('log', base=2)
#     plt.yscale('log', base=2)
#     plt.grid(True, which="both", ls="--", lw=0.5)
#     plt.legend(title="Baseline")
#     plt.tight_layout()

#     filename = f"plots/graph_density_{density}.png"
#     plt.savefig(filename, dpi=300)
#     plt.close()


# PRETTY PLOTS!

num_densities = len(densities)
cols = 2  # number of columns you want in the subplot grid
rows = (num_densities + cols - 1) // cols  # compute number of rows needed

cols = 1
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

# Flatten axes in case it's 2D
axes = axes.flatten()

for idx, distribution in enumerate(distributions):
    ax = axes[idx]
    
    for baseline in baselines:
        subset = df[(df['Density'] == 0.01) & (df['Baseline'] == baseline) & (df['Distribution'] == distribution)]
        subset = subset.sort_values('TasksPerNode')

        color, marker = baseline_style.get(baseline, ('black', 'o'))
        ax.plot(subset['TasksPerNode'], subset['Time'],
                marker=marker, color=color, label=baseline_labels.get(baseline, f"Baseline {baseline}"), markersize=5, linewidth=1.5)

    if distribution == 1:
        ax.set_title(f"{distribution_labels[distribution]} Distribution, Density = 0.01", fontsize=8)
    elif distribution == 2:
        ax.set_title(f"{distribution_labels[distribution]} Distribution ($\\lambda$ = 0.1), Density = 0.01", fontsize=8)


    ax.set_xlabel("Number of Processes", fontsize=8)
    ax.set_ylabel("Simulation Time (seconds)", fontsize=8)
    ax.set_xscale('log', base=2)
    # ax.set_yscale('log', base=2)
    # ax.grid(True, which="both", ls="--", lw=0.5)
    ax.tick_params(axis='both', which='major', labelsize=7)  # tick labels smaller
    ax.legend(title="Baseline", fontsize=6, title_fontsize=7, loc="best", frameon=True)

# Hide any unused subplots
for idx in range(len(distributions), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig("plots/combined_compressed_graphs.png", dpi=300)
plt.show()
