"""Run with: python -m wallboy --help"""
import argparse
import json
from pathlib import Path


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wallboy AI Rival: learn, watch, race.")
    commands = parser.add_subparsers(dest="command")
    train = commands.add_parser("train", help="Train a fresh DQN from random weights")
    from .settings import OPTIONS
    for option in OPTIONS:
        train.add_argument("--"+option.key.replace("_", "-"), type=type(option.default),
                           default=option.default, help=option.help)
    train.add_argument("--out", default="runs/my-first-run")
    train.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    train.add_argument("--live", action="store_true", help="Watch learning in Pygame; Tab toggles fast mode")
    for name in ("play", "watch", "race"):
        sub = commands.add_parser(name, help={"play":"Play manually", "watch":"Watch the greedy bot",
                                             "race":"Race your trained bot on identical courses"}[name])
        if name != "play":
            sub.add_argument("--model", default="runs/demo/best.pt")
        else:
            sub.add_argument("--classic", action="store_true", help="Original one-jump RL game")
            sub.add_argument("--nightmare", action="store_true", help="Faster challenge, up to 15 pixels/tick")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--endless", action="store_true")
        sub.add_argument("--frames", type=positive, help="Close after N frames (for smoke tests)")
        sub.add_argument("--screenshot", help="Save the last frame as a PNG")
    commands.add_parser("settings", help="Interactive RL experiment settings")
    commands.add_parser("graphs", help="Compare training runs on one graph")
    evaluate = commands.add_parser("evaluate", help="Test a policy on new seeded courses")
    evaluate.add_argument("--model", default="runs/demo/best.pt")
    evaluate.add_argument("--episodes", type=positive, default=50)
    evaluate.add_argument("--seed", type=int, default=30000)
    evaluate.add_argument("--json", help="Save full per-track results")
    plot = commands.add_parser("plot", help="Regenerate a reward graph from the CSV")
    plot.add_argument("--run", default="runs/demo")
    export = commands.add_parser("export", help="Export weights, C++ header and parity fixtures")
    export.add_argument("--model", default="runs/demo/best.pt")
    export.add_argument("--out", default="exports/demo")
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            from .ui import menu
            while True:
                selection = menu()
                if not selection:
                    break
                main(selection)
        elif args.command == "settings":
            from .lab_ui import training_settings
            selection = training_settings()
            if selection:
                main(selection)
        elif args.command == "graphs":
            from .lab_ui import graph_gallery
            graph_gallery()
        elif args.command == "train":
            from .train import train
            train(args)
        elif args.command in ("watch", "play", "race"):
            if args.command == "play" and not args.classic:
                from .challenge_ui import play_challenge
                play_challenge(args)
            else:
                from .ui import play
                play(args)
        elif args.command == "evaluate":
            import torch
            from .agent import load_policy
            from .evaluate import evaluate
            torch.set_num_threads(1)
            model, config, _ = load_policy(args.model)
            result = evaluate(model, config, range(args.seed, args.seed + args.episodes))
            print(json.dumps({k:v for k,v in result.items() if k != "results"}, indent=2))
            if args.json:
                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(result, indent=2))
        elif args.command == "plot":
            from .plot import plot_run
            print(plot_run(args.run))
        elif args.command == "export":
            from .export import export_policy
            export_policy(args.model, args.out)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Wallboy: {error}\n")


if __name__ == "__main__":
    main()
