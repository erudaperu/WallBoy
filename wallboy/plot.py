"""Create a portable PNG graph without requiring a desktop."""
import csv
import os
from pathlib import Path


def plot_run(folder):
    folder = Path(folder)
    os.environ.setdefault("MPLCONFIGDIR", str(folder.resolve() / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (folder / "episodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.set_facecolor("#0a0b18")
    for ax in axes:
        ax.set_facecolor("#111426")
        ax.grid(alpha=.13)
    if rows:
        episodes = [int(r["episode"]) for r in rows]
        rewards = [float(r["reward"]) for r in rows]
        moving = [sum(rewards[max(0, i-19):i+1]) / len(rewards[max(0, i-19):i+1])
                  for i in range(len(rewards))]
        axes[0].plot(episodes, rewards, color="#4d527b", alpha=.65, label="Training episode")
        axes[0].plot(episodes, moving, color="#55f4df", label="20-episode average")
        validation = [(int(r["episode"]), float(r["validation_reward"])) for r in rows
                      if r["validation_reward"]]
        if validation:
            x, y = zip(*validation)
            axes[0].plot(x, y, ".-", color="#ff65cf", label="Greedy validation (5 fixed tracks)")
        axes[1].plot(episodes, [float(r["cleared"]) for r in rows], color="#55f4df",
                     label="Obstacles cleared")
        twin = axes[1].twinx()
        twin.plot(episodes, [float(r["epsilon"]) for r in rows], color="#ffcc66", alpha=.8)
        twin.set_ylabel("Exploration probability", color="#ffcc66")
        twin.set_ylim(0, 1)
        axes[0].legend(loc="upper left", fontsize=8)
    else:
        axes[0].text(.5, .5, "Stopped before the first complete episode", ha="center",
                     transform=axes[0].transAxes)
    axes[0].set_title("WALLBOY / AI RIVAL — learning from collisions", loc="left", pad=18)
    axes[0].set_ylabel("Episode reward")
    axes[1].set_ylabel("Obstacles cleared")
    axes[1].set_xlabel("Completed training episode")
    fig.tight_layout()
    destination = folder / "learning.png"
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
