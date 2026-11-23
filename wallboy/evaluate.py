"""Greedy evaluation on courses separate from the training seeds."""
import random
import statistics

from .agent import greedy_action
from .env import Runner


def evaluate(model, config, seeds, random_actions=False):
    rows = []
    for seed in seeds:
        env = Runner(config)
        observation = env.reset(seed)
        rng = random.Random(seed + 1_000_000)
        while not env.done:
            action = rng.randrange(2) if random_actions else greedy_action(model, observation)
            observation, _, _, _ = env.step(action)
        rows.append({"seed": seed, "reward": round(env.total_reward, 4),
                     "cleared": env.cleared, "seconds": env.ticks / 60,
                     "survived_limit": env.truncated})
    return {"episodes": len(rows),
            "mean_reward": statistics.mean(r["reward"] for r in rows),
            "mean_cleared": statistics.mean(r["cleared"] for r in rows),
            "mean_seconds": statistics.mean(r["seconds"] for r in rows),
            "survival_rate": statistics.mean(r["survived_limit"] for r in rows),
            "results": rows}
