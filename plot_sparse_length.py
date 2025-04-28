import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("sparse-length.csv")

baseline_labels = {
    # 2: 'Naive AlltoAllv',
    # 3: 'Butterfly',
    # 4: 'Tree',
    11: 'Naive AlltoAllv',
    12: 'Delta Encoding AlltoAllv',
    13: 'Bitmap Encoding AlltoAllv',
}

# remove the rows from df that don't have a Baseline value in the baseline_labels
df = df[df['Baseline'].isin(baseline_labels.keys())]


density = 0.1
baselines = sorted(df['Baseline'].unique())
lengths = sorted(df['Length'].unique())

print(lengths)




colors = ['blue', 'orange', 'green', 'red', 'purple']
markers = ['o', 's', 'd', '^', '*']

baseline_style = {
    # 2: (colors[0], markers[0]),
    # 3: (colors[1], markers[1]),
    # 4: (colors[2], markers[2]),
    11: (colors[0], markers[0]),
    12: (colors[1], markers[1]),
    13: (colors[2], markers[2]),
}

os.makedirs("plots", exist_ok=True)

for length in lengths:
    plt.figure(figsize=(8, 6))
    
    for baseline in baselines:
        subset = df[(df['Length'] == length) & (df['Baseline'] == baseline)]
        subset = subset.sort_values('TasksPerNode')

        color, marker = baseline_style.get(baseline, ('black', 'o'))
        plt.plot(subset['TasksPerNode'], subset['Time'],
                 marker=marker, color=color, label=baseline_labels.get(baseline, f"Baseline {baseline}"))
    
    plt.title(f"Sparsity Complexity (Density = {density}, Length = {length})")
    plt.xlabel("Number of Processes")
    plt.xscale('log', base=2)
    plt.ylabel("Simulation Time (seconds)")
    # plt.yscale('log', base=2)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend(title="Baseline")
    plt.tight_layout()

    filename = f"plots/length_graph_compression_{length}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
