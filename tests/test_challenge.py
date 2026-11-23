"""Mechanics and course feasibility, not an AI player."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from wallboy.challenge import Challenge, Hazard, Orb, PATTERNS, make_pattern, MIN_GAP_TICKS


def empty_course():
    env = Challenge()
    env.obstacles = []
    env.orbs = []
    env.next_pattern = 100000
    return env


class ChallengeTests(unittest.TestCase):
    def test_keyboard_double_jump_and_fast_fall(self):
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        from wallboy import challenge_ui as ui
        captured = []
        original_screen = ui.Screen
        class Clock:
            def tick(self, fps):
                return 17
        class TestScreen(original_screen):
            def __init__(self):
                super().__init__()
                self.clock = Clock()
        def make_game(seed, nightmare):
            env = Challenge(seed, nightmare)
            captured.append(env)
            return env
        frame = 0
        def events():
            nonlocal frame
            frame += 1
            if frame in (1, 19, 21):
                key = ui.pg.K_DOWN if frame == 21 else ui.pg.K_SPACE
                return [ui.pg.event.Event(ui.pg.KEYDOWN, key=key)]
            return []
        args = SimpleNamespace(seed=42, nightmare=False, frames=22, screenshot=None)
        with patch.object(ui, "Screen", TestScreen), patch.object(ui, "Challenge", make_game), \
             patch.object(ui.pg.event, "get", events):
            ui.play_challenge(args)
        self.assertEqual(captured[0].jumps_left, 0)
        self.assertLess(captured[0].vy, -15)
        self.assertFalse(captured[0].dead)

    def test_two_jumps_no_third_and_landing_recharges(self):
        env = empty_course()
        env.tick(1)
        for _ in range(16):
            env.tick()
        env.tick(1)
        self.assertEqual(env.jumps_left, 0)
        previous_velocity = env.vy
        env.tick(1)
        self.assertLess(env.vy, previous_velocity)
        peak = env.y
        for _ in range(90):
            env.tick()
            peak = max(peak, env.y)
        self.assertGreater(peak, 190)
        self.assertEqual((env.y, env.jumps_left), (0, 2))

    def test_fast_fall(self):
        env = empty_course()
        env.tick(1)
        for _ in range(10):
            env.tick()
        height = env.y
        env.tick(2)
        self.assertLess(env.y, height)
        self.assertLess(env.vy, -15)

    def test_laser_safe_on_ground_lethal_in_air(self):
        grounded = empty_course()
        grounded.obstacles = [Hazard(140, 130, 18, bottom=54, kind="laser")]
        airborne = copy.deepcopy(grounded)
        airborne.y = 35
        grounded.tick()
        airborne.tick()
        self.assertFalse(grounded.dead)
        self.assertTrue(airborne.dead)

    def test_orb_and_clear_score_once(self):
        env = empty_course()
        env.orbs = [Orb(155, 18)]
        env.obstacles = [Hazard(110, 30, 20)]
        env.tick()
        self.assertEqual((env.collected, env.cleared, env.score), (1, 1, 150))
        env.tick()
        self.assertEqual(env.score, 150)

    def test_seeded_layout_does_not_depend_on_jumps(self):
        a, b = Challenge(9), Challenge(9)
        for i in range(30):
            a.tick(int(i in (2, 20)))
            b.tick()
        self.assertEqual(a.obstacles, b.obstacles)
        self.assertEqual(a.speed, b.speed)

    def test_speed_ramp_has_a_cap(self):
        for nightmare, start, cap in ((False, 7.8, 12), (True, 10, 15)):
            env = empty_course()
            env.nightmare = nightmare
            env.ticks = 898
            env.tick()
            self.assertEqual(env.level, 1)
            self.assertAlmostEqual(env.speed, start)
            env.tick()
            self.assertEqual(env.level, 2)
            self.assertAlmostEqual(env.speed, start + 1)
            self.assertGreater(env.level_up_until, env.ticks)
            env.tick()
            self.assertAlmostEqual(env.speed, start + 1)
            env.ticks = 100000
            env.tick()
            self.assertEqual(env.speed, cap)
            env.reset(42)
            self.assertEqual(env.level, 1)
            self.assertAlmostEqual(env.speed, start)
            self.assertEqual(env.level_up_until, 0)

    def test_all_patterns_are_possible_at_start_and_top_speeds(self):
        # Exhaustively try two press times on an ISOLATED pattern. This is a
        # design check only: these schedules never control a player in the game.
        patterns = [make_pattern(kind, 400) for kind in PATTERNS]
        for nightmare in (False, True):
            for elapsed in (0, 6000):
                for pattern in patterns:
                    success = False
                    for first in range(-1, 80, 2):
                        if success:
                            break
                        for delay in range(4, 34, 2):
                            env = empty_course()
                            env.nightmare = nightmare
                            env.ticks = elapsed
                            env.obstacles = copy.deepcopy(pattern)
                            # Match the actual minimum initial warning distance.
                            for hazard in env.obstacles:
                                hazard.x += 280
                            for tick in range(140):
                                env.tick(int(first >= 0 and tick in (first, first + delay)))
                                if env.dead or env.cleared == len(pattern):
                                    break
                            if not env.dead and env.cleared == len(pattern):
                                success = True
                                break
                    self.assertTrue(success, (nightmare, elapsed, pattern))

    def test_density_tightens_but_keeps_recovery_space(self):
        for nightmare in (False, True):
            gaps = []
            for elapsed in (0, 3600, 100000):
                env = empty_course()
                env.nightmare = nightmare
                env.ticks = elapsed
                env.next_pattern = 1800  # Generate exactly one group.
                env._fill_track()
                end = env.obstacles[-1].x + env.obstacles[-1].width
                gaps.append(env.next_pattern - end)
            self.assertGreater(gaps[0], gaps[1])
            self.assertEqual(gaps[1], gaps[2])
            self.assertGreaterEqual(gaps[1], (15 if nightmare else 12) * MIN_GAP_TICKS)


if __name__ == "__main__":
    unittest.main()
