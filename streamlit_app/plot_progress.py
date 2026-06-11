import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Generating Week 4 Presentation Slide...")

# Accuracy evolution over the 4 phases
weeks = ['Week 1\n(Baseline)', 'Week 1\n(Cleaned)', 'Week 2\n(Feature Fusion)', 'Week 3\n(DistilBERT)']
accuracies = [98.74, 99.11, 99.38, 99.61]

# Set up the visual style
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(weeks, accuracies, color=['#9B9B9B', '#4A90E2', '#50E3C2', '#B8E986'])

# Polish the chart
ax.set_ylim(98.0, 100.0)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Evolution (4 Weeks)', fontsize=16, fontweight='bold', pad=20)

# Add value labels above the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_progression.png', dpi=200)
print("Saved to accuracy_progression.png!")
