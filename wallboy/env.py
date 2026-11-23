"""The entire game, independent of PyTorch and Pygame.

Coordinates are pixels. y is the height of the player's FEET above the floor.
Physics runs at 60 Hz; one agent decision advances four physics ticks.
"""
from dataclasses import asdict, dataclass
import random


@dataclass(frozen=True)
class Config:
    width: int = 960
    player_x: float = 140.0
    player_width: float = 26.0
    player_height: float = 34.0
    gravity: float = 0.65
    jump_velocity: float = 11.5
    action_ticks: int = 4
    max_ticks: int = 1800  # Training/evaluation time limit; 0 means endless.
    speed_min: float = 5.0
    speed_max: float = 7.0
    gap_min: float = 320.0
    gap_max: float = 540.0
    alive_reward: float = .02
    clear_reward: float = 3.0
    jump_cost: float = .03
    death_cost: float = 5.0


OBSERVATIONS = ["distance / 600", "obstacle width / 80", "obstacle height / 100",
                "feet height / 120", "vertical velocity / 15", "on ground",
                "scroll speed / 10"]


@dataclass
class Obstacle:
    x: float
    width: float
    height: float
    passed: bool = False


class Runner:
    def __init__(self, config=None):
        self.config = config or Config()
        self.reset(0)

    def reset(self, seed):
        self.rng = random.Random(seed)  # Never touches the agent's RNG.
        self.speed = self.rng.uniform(self.config.speed_min, self.config.speed_max)
        self.y = self.vy = self.total_reward = 0.0
        self.ticks = self.cleared = 0
        self.dead = self.truncated = False
        self.obstacles = []
        self._add_obstacle(500.0)
        self._fill_track()
        return self.observe()

    def _add_obstacle(self, x):
        self.obstacles.append(Obstacle(x, self.rng.uniform(22, 44),
                                      self.rng.uniform(28, 44)))

    def _fill_track(self):
        while self.obstacles[-1].x < self.config.width + 600:
            last = self.obstacles[-1]
            x = last.x + last.width + self.rng.uniform(self.config.gap_min,
                                                      self.config.gap_max)
            self._add_obstacle(x)

    @property
    def done(self):
        return self.dead or self.truncated

    def observe(self):
        c = self.config
        obstacle = next(o for o in self.obstacles if o.x + o.width > c.player_x)
        # Signed distance is to the player's FRONT edge. Do not clip at zero.
        return [(obstacle.x - c.player_x - c.player_width) / 600,
                obstacle.width / 80, obstacle.height / 100,
                self.y / 120, self.vy / 15, float(self.y == 0), self.speed / 10]

    def tick(self, action):
        """One physics tick. Jump commands while airborne have no effect."""
        if action not in (0, 1):
            raise ValueError("action must be 0 (wait) or 1 (jump)")
        if self.done:
            raise RuntimeError("Reset the finished episode before stepping")
        c = self.config
        reward = c.alive_reward
        if action == 1 and self.y == 0:
            self.vy = c.jump_velocity
            reward -= c.jump_cost
        self.y = max(0.0, self.y + self.vy)
        self.vy = self.vy - c.gravity if self.y > 0 else 0.0
        self.ticks += 1
        for obstacle in self.obstacles:
            obstacle.x -= self.speed
            overlaps = (obstacle.x < c.player_x + c.player_width and
                        obstacle.x + obstacle.width > c.player_x)
            if overlaps and self.y < obstacle.height:
                self.dead = True
            if not obstacle.passed and obstacle.x + obstacle.width <= c.player_x:
                obstacle.passed = True
                self.cleared += 1
                reward += c.clear_reward
        if self.dead:
            reward -= c.death_cost
        self.obstacles = [o for o in self.obstacles if o.x + o.width > -40]
        self._fill_track()
        self.truncated = bool(c.max_ticks and self.ticks >= c.max_ticks and not self.dead)
        self.total_reward += reward
        return reward

    def step(self, action):
        """Return observation, reward, terminated, truncated.

        A jump is a button press at the start of a decision, not an auto-jump
        when landing later in the four-tick window.
        """
        reward = 0.0
        for tick in range(self.config.action_ticks):
            reward += self.tick(action if tick == 0 else 0)
            if self.done:
                break
        return self.observe(), reward, self.dead, self.truncated

    def metadata(self):
        return asdict(self.config)
