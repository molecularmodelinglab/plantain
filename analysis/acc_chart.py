import matplotlib.pyplot as plt
import numpy as np

programs = ['Vina', 'GNINA', "Twister"]
colors = [ "#0099cc", "#0099cc", "#ff6666" ]
acc_2 = [14.2, 12.8, 25.8 ]
# acc_5 = [45, 40.6, 69]

# Calculate x-coordinates for top-5 bars
bar_width = 0.35
x = np.arange(len(programs))
# x_top5 = x + bar_width*0.5

# Create bar chart for top-5 accuracies
fig, ax = plt.subplots(figsize=(10, 5))
# ax.bar(x_top5, acc_5, color=colors, alpha=0.1, width=bar_width)

# Create bar chart for top-2 accuracies
ax.bar(programs, acc_2, color=colors, width=bar_width)

# Set x-tick labels
# ax.set_xticks(x_top5)
# ax.set_xticklabels(programs, fontsize=14)

# Set chart title and axis labels
ax.set_ylabel('2 Å Accuracy (%)', fontsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('left')

plt.savefig('outputs/acc_chart.png', dpi=300)