import matplotlib.pyplot as plt

# Data for the blue line chart
x_labels = ["top 10", "top 20", "top 30", "top 100"]
y_values = [0.308, 0.290, 0.285, 0.328]

# Red line value
red_line_value = 0.34

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x_labels, y_values, color='blue', label='Hindsight Relabelling', marker='o')
plt.axhline(y=red_line_value, color='red', linestyle='--', label='Ground Truth Clustering')

# Label axes
plt.ylabel('Success Rate')
plt.xlabel('Top-k')

# Add a title (optional)
# plt.title('Success Rate Comparison')

# Add legend
plt.legend()
plt.ylim(0.25, 0.4)

# Show the plot
plt.tight_layout()
# plt.show()

plt.savefig('plot.png')
