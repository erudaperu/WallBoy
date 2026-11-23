"""Collect experience, improve the Q network, and measure actual progress."""
import csv
import json
from pathlib import Path
import time

import torch

from .agent import DQN, load_policy, save_policy
from .env import Config, Runner
from .evaluate import evaluate
from .settings import validate, exploration


def train(args):
    torch.set_num_threads(1)  # Tiny matrix multiplies need very few CPU threads.
    settings = validate(vars(args))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty. Choose a new --out folder to keep previous runs.")
    out.mkdir(parents=True, exist_ok=True)
    config = Config(max_ticks=settings["max_ticks"], **{k: settings[k] for k in
                    ("alive_reward", "clear_reward", "jump_cost", "death_cost")})
    env = Runner(config)
    agent = DQN(args.seed, args.device, **{k:v for k,v in settings.items() if k != "seed"})
    validation_seeds = range(10_000, 10_005)
    save_policy(out / "initial.pt", agent.online, env.metadata(), episode=0, seed=args.seed)
    best = evaluate(agent.online, config, validation_seeds)["mean_reward"]
    save_policy(out / "best.pt", agent.online, env.metadata(), episode=0, seed=args.seed)
    (out / "settings.json").write_text(json.dumps({"arguments": vars(args),
        "environment": env.metadata(), "device_used": str(agent.device),
        "torch_version": str(torch.__version__), "validation_seeds": list(validation_seeds),
        **settings}, indent=2))
    view = None
    if args.live:
        from .ui import TrainingView
        view = TrainingView()
    rewards = []
    steps = completed = 0
    last_loss = None
    stopped = False
    start = time.monotonic()
    print(f"Training random weights on {agent.device}. Output: {out}", flush=True)
    try:
        with (out / "episodes.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "steps", "reward", "cleared", "seconds", "epsilon",
                             "loss", "validation_reward", "wall_seconds"])
            for episode in range(1, args.episodes + 1):
                # Training, checkpoint selection, and final test tracks never overlap.
                observation = env.reset(1_000_000 + args.seed * 100_000 + episode)
                while not env.done:
                    epsilon = exploration(steps, settings)
                    action, exploratory = agent.act(observation, epsilon)
                    next_observation, reward, terminated, _ = env.step(action)
                    agent.memory.append((observation, action, reward, next_observation, terminated))
                    loss = agent.learn()
                    if loss is not None:
                        last_loss = loss
                    observation = next_observation
                    steps += 1
                    if view and not view.update(env, agent.online, episode, steps, epsilon,
                                                last_loss, rewards, action, exploratory):
                        stopped = True
                        break
                if stopped:
                    break
                rewards.append(env.total_reward)
                completed = episode
                validation = None
                if episode % args.eval_every == 0 or episode == args.episodes:
                    validation = evaluate(agent.online, config, validation_seeds)["mean_reward"]
                    if validation > best:
                        best = validation
                        save_policy(out / "best.pt", agent.online, env.metadata(),
                                    episode=episode, seed=args.seed, validation_reward=best)
                    save_policy(out / "last.pt", agent.online, env.metadata(),
                                episode=episode, seed=args.seed)
                writer.writerow([episode, steps, env.total_reward, env.cleared, env.ticks / 60,
                                 epsilon, last_loss, validation, time.monotonic() - start])
                handle.flush()
                if episode % 10 == 0 or episode == 1:
                    print(f"ep {episode:4d} | decisions {steps:7d} | reward {env.total_reward:7.2f}"
                          f" | cleared {env.cleared:2d} | explore {epsilon:.0%}"
                          f" | best validation {best:.1f}", flush=True)
    except KeyboardInterrupt:
        stopped = True
        print("\nStopping and saving the latest policy...", flush=True)
    finally:
        save_policy(out / "last.pt", agent.online, env.metadata(), episode=completed,
                    seed=args.seed, steps=steps, interrupted=stopped)
        if view:
            view.close()
    # Held-out evaluation is performed after model selection, never used to select it.
    report = {"completed_episodes": completed, "decisions": steps,
              "wall_seconds": round(time.monotonic() - start, 2), "interrupted": stopped,
              "test_seeds": list(range(20_000, 20_020)), "policies": {}}
    initial, _, _ = load_policy(out / "initial.pt")
    report["policies"]["random_actions"] = evaluate(initial, config, report["test_seeds"], True)
    for name in ("initial", "best", "last"):
        model, _, metadata = load_policy(out / f"{name}.pt")
        report["policies"][name] = evaluate(model, config, report["test_seeds"])
        report["policies"][name]["checkpoint_metadata"] = metadata
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    from .plot import plot_run
    plot_run(out)
    print("Held-out mean obstacles cleared:", flush=True)
    for name, result in report["policies"].items():
        print(f"  {name:14s} {result['mean_cleared']:6.2f} | survives {config.max_ticks/60:g}s cap: "
              f"{result['survival_rate']:.0%}", flush=True)
    print(f"Saved policies, CSV, graph, and evaluation to {out}", flush=True)
