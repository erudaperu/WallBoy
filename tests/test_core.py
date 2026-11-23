"""Tests for gameplay, the RL update, and checkpoint portability."""
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from wallboy.agent import DQN, load_policy, q_values, save_policy
from wallboy.env import Config, Obstacle, Runner


class GameTests(unittest.TestCase):
    def test_equal_seeds_give_equal_tracks_despite_different_actions(self):
        a, b = Runner(), Runner()
        a.reset(19)
        b.reset(19)
        for tick in range(30):
            a.tick(int(tick == 0))
            b.tick(0)
        self.assertEqual(a.obstacles, b.obstacles)
        self.assertEqual(a.speed, b.speed)
        self.assertNotEqual(a.y, b.y)

    def test_jump_and_no_air_jump(self):
        a, b = Runner(), Runner()
        a.tick(1)
        b.tick(1)
        for _ in range(10):
            a.tick(1)
            b.tick(0)
        self.assertEqual((a.y, a.vy), (b.y, b.vy))
        for _ in range(30):
            a.tick(0)
        self.assertEqual((a.y, a.vy), (0, 0))

    def test_collision_and_pass_rewards(self):
        env = Runner()
        env.obstacles = [Obstacle(150, 30, 40), Obstacle(700, 30, 40)]
        reward = env.tick(0)
        self.assertTrue(env.dead)
        self.assertAlmostEqual(reward, -4.98)
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(5)
        env.y = 70
        env.obstacles = [Obstacle(110, 30, 40), Obstacle(700, 30, 40)]
        self.assertAlmostEqual(env.tick(0), 3.02)
        self.assertEqual(env.cleared, 1)
        self.assertFalse(env.dead)

    def test_time_limit_is_not_a_collision(self):
        env = Runner(replace(Config(), max_ticks=4))
        _, _, terminated, truncated = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(env.ticks, 4)

    def test_playback_physics_equals_training_physics(self):
        a, b = Runner(), Runner()
        a.reset(123)
        b.reset(123)
        rng = random.Random(4)
        while not a.done:
            action = rng.randrange(2)
            a.step(action)
            for tick in range(b.config.action_ticks):
                b.tick(action if tick == 0 else 0)
                if b.done:
                    break
            self.assertEqual(a.observe(), b.observe())
            self.assertEqual((a.total_reward, a.cleared, a.dead),
                             (b.total_reward, b.cleared, b.dead))


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_experience_changes_weights_and_roundtrips(self):
        agent, env = DQN(), Runner()
        before = [p.detach().clone() for p in agent.online.parameters()]
        obs = env.observe()
        for step in range(1005):
            action, _ = agent.act(obs, 1.0)
            following, reward, terminated, _ = env.step(action)
            agent.memory.append((obs, action, reward, following, terminated))
            obs = env.reset(step+1) if env.done else following
        loss = agent.learn()
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a,b in zip(before, agent.online.parameters())))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.pt"
            save_policy(path, agent.online, env.metadata(), episode=1)
            loaded, config, metadata = load_policy(path)
            np.testing.assert_array_equal(q_values(loaded, obs), q_values(agent.online, obs))
            self.assertEqual(config, env.config)
            self.assertEqual(metadata["episode"], 1)

    def test_collisions_stop_bootstrap_but_time_limits_do_not(self):
        def loss_for(terminated):
            agent = DQN()
            with torch.no_grad():
                for model in (agent.online, agent.target):
                    for parameter in model.parameters():
                        parameter.zero_()
                    model.layers[-1].bias.fill_(2)
            obs = Runner().observe()
            agent.memory.extend([(obs, 0, 0.0, obs, terminated)] * 1000)
            return agent.learn()
        self.assertAlmostEqual(loss_for(True), 1.5, places=5)
        self.assertAlmostEqual(loss_for(False), .0002, places=5)


if __name__ == "__main__":
    unittest.main()
