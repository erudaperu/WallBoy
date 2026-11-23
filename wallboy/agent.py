"""A readable DQN: neural Q values, experience replay, and a target network."""
from collections import deque
import copy
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .env import Config, OBSERVATIONS
from .settings import validate


class QNetwork(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(nn.Linear(7, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2))

    def forward(self, observation):
        return self.layers(observation)


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu; no driver changes are needed.")
    return torch.device(name)


@torch.no_grad()
def q_values(model, observation):
    device = next(model.parameters()).device
    x = torch.tensor(observation, dtype=torch.float32, device=device)
    return model(x).cpu().numpy()


def greedy_action(model, observation):
    return int(q_values(model, observation).argmax())


class DQN:
    def __init__(self, seed=7, device="cpu", **settings):
        self.settings = validate(settings)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.device = choose_device(device)
        self.online = QNetwork(self.settings["hidden_size"]).to(self.device)  # RANDOM initial weights.
        self.target = copy.deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.settings["learning_rate"])
        self.memory = deque(maxlen=self.settings["replay_size"])
        self.updates = 0

    def act(self, observation, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(2), True
        return greedy_action(self.online, observation), False

    def learn(self):
        if len(self.memory) < max(self.settings["warmup_steps"], self.settings["batch_size"]):
            return None
        batch = self.rng.sample(self.memory, self.settings["batch_size"])
        states, actions, rewards, next_states, terminated = zip(*batch)
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(terminated, dtype=torch.float32, device=self.device)
        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Time limits still bootstrap; a collision really ends the future.
            future = self.target(next_states).max(dim=1).values
            desired = rewards + self.settings["gamma"] * (1 - terminated) * future
        loss = nn.functional.smooth_l1_loss(predicted, desired)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.settings["gradient_clip"])
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.settings["target_update"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())


def save_policy(path, model, config, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"format_version": 1, "hidden_size": model.hidden_size, "state_dict": model.state_dict(),
                "config": config, "observations": OBSERVATIONS,
                "metadata": metadata}, temporary)
    temporary.replace(path)


def load_policy(path, device="cpu"):
    device = choose_device(device)
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format_version") != 1 or data.get("observations") != OBSERVATIONS:
        raise ValueError("This checkpoint uses a different observation format")
    model = QNetwork(data.get("hidden_size", 64)).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, Config(**data["config"]), data["metadata"]
