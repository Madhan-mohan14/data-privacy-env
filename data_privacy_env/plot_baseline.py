"""Generate reward baseline plot from baseline_results.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("baseline_results.json") as f:
    data = json.load(f)

results = data["results"]
episodes = [r["episode"] for r in results]
rewards = [r["reward"] for r in results]
levels = [r["level"] for r in results]

colors = {1: "#2196F3", 3: "#FF9800"}
point_colors = [colors[l] for l in levels]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"ComplianceGuard Baseline — {data['model']}\n"
    f"30 episodes | Success rate: {data['success_rate']:.1%} | Avg reward: {data['avg_reward']:.3f}",
    fontsize=13, fontweight="bold",
)

# Left: episode rewards scatter + line
l1_eps = [r["episode"] for r in results if r["level"] == 1]
l1_rew = [r["reward"] for r in results if r["level"] == 1]
l3_eps = [r["episode"] for r in results if r["level"] == 3]
l3_rew = [r["reward"] for r in results if r["level"] == 3]

ax1.plot(l1_eps, l1_rew, "o-", color="#2196F3", label="Level 1 (simple logs)", linewidth=1.5, markersize=6)
ax1.plot(l3_eps, l3_rew, "s-", color="#FF9800", label="Level 3 (obfuscated PII)", linewidth=1.5, markersize=6)
ax1.axhline(0.7, color="#4CAF50", linestyle="--", linewidth=1.5, label="Success threshold (0.7)")
ax1.axhline(0.05, color="#F44336", linestyle=":", linewidth=1, label="Floor reward (0.05)")
ax1.set_xlabel("Episode", fontsize=11)
ax1.set_ylabel("Final Episode Reward", fontsize=11)
ax1.set_title("Episode Rewards by Level", fontsize=12)
ax1.legend(fontsize=9)
ax1.set_ylim(-0.05, 1.1)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor("#FAFAFA")

# Right: bar chart — success rate by level
l1_sr = sum(1 for r in results if r["level"] == 1 and r["success"]) / max(1, len([r for r in results if r["level"] == 1]))
l3_sr = sum(1 for r in results if r["level"] == 3 and r["success"]) / max(1, len([r for r in results if r["level"] == 3]))
overall_sr = data["success_rate"]

bars = ax2.bar(
    ["Level 1\n(easy)", "Level 3\n(hard)", "Overall"],
    [l1_sr, l3_sr, overall_sr],
    color=["#2196F3", "#FF9800", "#9C27B0"],
    width=0.5, alpha=0.85,
)
ax2.axhline(0.3, color="#4CAF50", linestyle="--", linewidth=1.5, label="GREEN gate (30%)")
for bar, val in zip(bars, [l1_sr, l3_sr, overall_sr]):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.0%}", ha="center", fontsize=12, fontweight="bold")
ax2.set_ylabel("Success Rate (reward ≥ 0.7)", fontsize=11)
ax2.set_title("Success Rate by Difficulty Level", fontsize=12)
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_facecolor("#FAFAFA")

gate_color = "#4CAF50" if data["gate"] == "GREEN" else ("#FF9800" if data["gate"] == "YELLOW" else "#F44336")
fig.text(0.5, 0.01, f"Training gate: {data['gate']} — Baseline clears the 30% threshold for GRPO training",
         ha="center", fontsize=10, color=gate_color, fontweight="bold")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("assets/baseline_reward_curve.png", dpi=150, bbox_inches="tight")
print("Plot saved: assets/baseline_reward_curve.png")
