import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Data
data = {
    "Model only": 0.99,
    "System prompting": 1.00,
    "Supervised fine-tuning": 0.32,
    "Refusal vector removal": 1.00,
}

# Extract keys and values
methods = list(data.keys())
rates = list(data.values())

# Use a blue colormap for nicer aesthetics
colors = cm.Blues(np.linspace(0.5, 0.9, len(methods)))

# Plot
plt.figure(figsize=(9, 6))
bars = plt.bar(methods, rates, color=colors, alpha=0.9,
               edgecolor="black", linewidth=0.8)

# Annotate bars with percentage labels
for bar, rate in zip(bars, rates):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{rate*100:.0f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#1f2d4d"
    )

# Title and labels
plt.title("Refusal Rate of GPT-OSS-20B Across Uncensoring Techniques",
          fontsize=16, fontweight="bold", color="#1f2d4d")
plt.ylabel("Refusal Rate", fontsize=13, color="#1f2d4d")
plt.ylim(0, 1.15)

# Style tweaks
plt.xticks(rotation=15, ha="right", fontsize=12, color="#1f2d4d")
plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%",
           "75%", "100%"], fontsize=11, color="#1f2d4d")
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Minimalist frame
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()
plt.show()
plt.savefig("./plot.png")
