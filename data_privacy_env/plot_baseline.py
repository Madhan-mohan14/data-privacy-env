"""Generate baseline performance plots from baseline_results.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("baseline_results.json") as f:
    data = json.load(f)

results = data["results"]
LEVEL_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 4: "#F44336"}
LEVEL_LABELS = {
    1: "L1 — clear logs",
    2: "L2 — multi-file",
    3: "L3 — obfuscated PII",
    4: "L4 — red herrings",
}

# ── per-level stats ──────────────────────────────────────────────────────────
by_level = {}
for lvl in [1, 2, 3, 4]:
    lvl_results = [r for r in results if r["level"] == lvl]
    if not lvl_results:
        continue
    avg = sum(r["reward"] for r in lvl_results) / len(lvl_results)
    sr  = sum(1 for r in lvl_results if r["success"]) / len(lvl_results)
    by_level[lvl] = {"avg": avg, "sr": sr, "results": lvl_results}

# ── Figure 1: reward by level (bar) + per-episode scatter ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"ComplianceGuard Baseline — {data['model']}\n"
    f"{data['n_episodes']} episodes | Avg reward: {data['avg_reward']:.3f} | "
    f"Gate: {data['gate']}",
    fontsize=13, fontweight="bold",
)

# Left: bar chart — avg reward by level
ax = axes[0]
lvls = sorted(by_level.keys())
avgs = [by_level[l]["avg"] for l in lvls]
bars = ax.bar(
    [LEVEL_LABELS[l] for l in lvls],
    avgs,
    color=[LEVEL_COLORS[l] for l in lvls],
    width=0.55, alpha=0.88, edgecolor="white", linewidth=1.5,
)
ax.axhline(0.70, color="#388E3C", linestyle="--", linewidth=1.8, label="Success threshold (0.70)")
ax.axhline(0.05, color="#B71C1C", linestyle=":", linewidth=1.2, label="Floor reward (0.05)")
for bar, val in zip(bars, avgs):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Average Episode Reward", fontsize=11)
ax.set_title("Baseline Reward by Curriculum Level", fontsize=12)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#FAFAFA")

# Right: per-episode rewards (all 4 levels)
ax = axes[1]
for lvl in sorted(by_level.keys()):
    eps = [r["episode"] for r in by_level[lvl]["results"]]
    rws = [r["reward"] for r in by_level[lvl]["results"]]
    ax.plot(eps, rws, "o-", color=LEVEL_COLORS[lvl],
            label=LEVEL_LABELS[lvl], linewidth=1.6, markersize=7)
ax.axhline(0.70, color="#388E3C", linestyle="--", linewidth=1.8, label="Success threshold")
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Final Episode Reward", fontsize=11)
ax.set_title("Per-Episode Rewards (all levels)", fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.3)
ax.set_facecolor("#FAFAFA")

gate_color = "#2E7D32" if data["gate"] == "GREEN" else ("#E65100" if data["gate"] == "YELLOW" else "#C62828")
fig.text(0.5, 0.01,
         f"Training gate: {data['gate']} — L1 partially solved; L3/L4 unsolved → clear GRPO training target",
         ha="center", fontsize=10, color=gate_color, fontweight="bold")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("assets/baseline_reward_curve.png", dpi=150, bbox_inches="tight")
print("Saved: assets/baseline_reward_curve.png")

# ── Figure 2: learning gap — baseline vs expected post-training ───────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
fig2.suptitle("ComplianceGuard — Learning Gap (Baseline vs Training Target)",
              fontsize=13, fontweight="bold")

baseline_avgs = [by_level[l]["avg"] for l in [1, 2, 3, 4]]
# Conservative post-training targets based on GRPO on similar tasks
target_avgs  = [0.90, 0.75, 0.68, 0.60]

x = np.arange(4)
w = 0.35
bars1 = ax2.bar(x - w/2, baseline_avgs, w, label="Baseline (Qwen2.5-7B, no training)",
                color="#78909C", alpha=0.85, edgecolor="white")
bars2 = ax2.bar(x + w/2, target_avgs, w, label="Target (post GRPO, Qwen2.5-1.5B)",
                color="#1565C0", alpha=0.85, edgecolor="white")

for bar, val in zip(bars1, baseline_avgs):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f"{val:.2f}", ha="center", fontsize=10, color="#37474F")
for bar, val in zip(bars2, target_avgs):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f"{val:.2f}", ha="center", fontsize=10, color="#0D47A1")

ax2.axhline(0.70, color="#388E3C", linestyle="--", linewidth=1.8, label="Success threshold (0.70)")
ax2.set_xticks(x)
ax2.set_xticklabels([LEVEL_LABELS[l] for l in [1, 2, 3, 4]], fontsize=10)
ax2.set_ylabel("Average Reward", fontsize=11)
ax2.set_title("Baseline vs Training Target (post-GRPO estimates)", fontsize=11)
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_facecolor("#FAFAFA")
fig2.text(0.5, 0.01, "Targets are projections — actual results will be updated after training run.",
          ha="center", fontsize=9, color="#757575", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("assets/learning_gap.png", dpi=150, bbox_inches="tight")
print("Saved: assets/learning_gap.png")
