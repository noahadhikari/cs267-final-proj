import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the file
df = pd.read_csv("data/hw2-2strong.csv")

# Extract data
x = df.iloc[:, 0].values  # Number of threads
y_values = [df.iloc[:, i].values for i in range(1, df.shape[1])]
threads = df.columns[1:]

# Plot
plt.figure(figsize=(8, 6))
for i, y in enumerate(y_values):
    plt.loglog(x, y, marker='o', label=threads[i])

# Add reference line (slope = 1)
x_ref = np.array([min(x), max(x)])
y_ref = x_ref[0] / x_ref * y_values[0][0]  # Scale to first data point and invert slope
plt.loglog(x_ref, y_ref, 'k--', label='Slope = -1')

# Labels and legend
plt.xlabel('Number of Ranks')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.savefig("plots/strong_scaling.png")
