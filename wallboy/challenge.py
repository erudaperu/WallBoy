"""Manual challenge mode. Read tick() to learn how double jumping works.

This game deliberately has no bot or learning code. Classic RL remains in env.py.
"""
from dataclasses import dataclass
import random

from .env import Config

LEVEL_SECONDS = 15
TICKS_PER_LEVEL = 60 * LEVEL_SECONDS
# Even at maximum speed, reserve time to land and prepare the next jump.
MIN_GAP_TICKS = 55


@dataclass
class Hazard:
    x: float
    width: float
    height: float
    bottom: float = 0
    kind: str = "barrier"
    passed: bool = False


@dataclass
class Orb:
    x: float
    y: float
    collected: bool = False


# Each tuple is (horizontal offset, width, height, height above the floor).
# Keep each group short enough to clear with the same two-jump allowance.
PATTERNS = {
    "tower": [(0, 54, 142, 0)],
    "pair": [(0, 36, 48, 0), (190, 42, 62, 0)],
    "laser": [(0, 130, 18, 54)],
    "wide": [(0, 124, 68, 0)],
    "triple": [(0, 28, 32, 0), (130, 28, 48, 0), (260, 28, 32, 0)],
    "stairs": [(0, 32, 48, 0), (110, 32, 100, 0), (220, 42, 142, 0)],
}


def make_pattern(kind, x):
    return [Hazard(x + offset, width, height, bottom, kind)
            for offset, width, height, bottom in PATTERNS[kind]]


class Challenge:
    def __init__(self, seed=42, nightmare=False):
        self.config = Config(max_ticks=0)
        self.nightmare = nightmare
        self.reset(seed)

    def reset(self, seed):
        self.seed = seed
        self.rng = random.Random(seed)
        self.ticks = self.cleared = self.score = self.collected = 0
        self.y = self.vy = self.distance = 0.0
        self.jumps_left = 2
        self.dead = False
        self.level_up_until = 0
        self.speed = 10.0 if self.nightmare else 7.8
        self.obstacles, self.orbs = [], []
        self.next_pattern = 680.0
        self.last_pattern = None
        self.notice = "TAP TWICE TO DOUBLE JUMP"
        self.notice_until = 180
        self._fill_track()

    @property
    def done(self):
        return self.dead

    @property
    def multiplier(self):
        return min(5, 1 + self.cleared // 5)

    @property
    def level(self):
        return 1 + self.ticks // TICKS_PER_LEVEL

    def _fill_track(self):
        while self.next_pattern < 1900:
            # All patterns have a clear landing zone before the next pattern.
            choices = [p for p in PATTERNS if p != self.last_pattern]
            kind = self.rng.choice(choices)
            x = self.next_pattern
            hazards = make_pattern(kind, x)
            self.obstacles.extend(hazards)
            end = hazards[-1].x + hazards[-1].width
            orb_height = 18 if kind == "laser" else 192 if kind in ("tower", "stairs") else 130
            self.orbs.append(Orb((x + end) / 2, orb_height))
            self.last_pattern = kind
            # Density rises until level 5. Use the MAX speed, not current speed,
            # so a later level-up cannot consume the reserved recovery time.
            minimum_gap = (15 if self.nightmare else 12) * MIN_GAP_TICKS
            gap = minimum_gap + max(0, 5 - self.level) * 20
            self.next_pattern = end + self.rng.uniform(gap, gap + 120)

    def tick(self, action=0):
        """0 wait, 1 jump button press, 2 fast fall. Holding jump never repeats."""
        if action not in (0, 1, 2):
            raise ValueError("Challenge actions: 0 wait, 1 jump, 2 fast fall")
        if self.dead:
            return
        c = self.config
        if action == 1 and self.jumps_left:
            # The second press resets upward velocity, even while falling.
            self.vy = c.jump_velocity
            self.jumps_left -= 1
            self.notice = "DOUBLE JUMP!" if self.jumps_left == 0 else "JUMP"
            self.notice_until = self.ticks + 30
        elif action == 2 and self.y > 0:
            self.vy = -16.0
        self.y = max(0.0, self.y + self.vy)
        self.vy = self.vy - c.gravity if self.y > 0 else 0.0
        if self.y == 0:
            self.jumps_left = 2  # Landing recharges both jumps.
        previous_level = self.level
        self.ticks += 1
        # Each 15 seconds survived earns a level and a discrete speed increase.
        self.speed = min(15.0 if self.nightmare else 12.0,
                         (10.0 if self.nightmare else 7.8) + self.level - 1)
        if self.level > previous_level:
            self.level_up_until = self.ticks + 120
        self.distance += self.speed
        self.next_pattern -= self.speed
        for hazard in self.obstacles:
            hazard.x -= self.speed
            overlap_x = hazard.x < c.player_x + c.player_width and hazard.x + hazard.width > c.player_x
            overlap_y = self.y < hazard.bottom + hazard.height and self.y + c.player_height > hazard.bottom
            if overlap_x and overlap_y:
                self.dead = True
            if not hazard.passed and hazard.x + hazard.width <= c.player_x:
                hazard.passed = True
                self.score += 100 * self.multiplier
                self.cleared += 1
        for orb in self.orbs:
            orb.x -= self.speed
            if (not self.dead and not orb.collected and
                c.player_x - 8 < orb.x < c.player_x + c.player_width + 8 and
                self.y - 8 < orb.y < self.y + c.player_height + 8):
                orb.collected = True
                self.collected += 1
                self.score += 50 * self.multiplier
                self.notice, self.notice_until = "+ ENERGY", self.ticks + 30
        self.obstacles = [h for h in self.obstacles if h.x + h.width > -40]
        self.orbs = [o for o in self.orbs if o.x > -40 and not o.collected]
        self._fill_track()
