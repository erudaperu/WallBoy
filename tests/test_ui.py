"""Exercise real Pygame events and rendering through SDL's headless driver."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from wallboy import ui
from wallboy.agent import QNetwork
from wallboy.env import Runner


class UITests(unittest.TestCase):
    def test_menu_keyboard_selection(self):
        original = ui.Screen
        class SelectWatch(original):
            def __init__(self):
                super().__init__()
                for key in (ui.pg.K_DOWN, ui.pg.K_RETURN):
                    ui.pg.event.post(ui.pg.event.Event(ui.pg.KEYDOWN, key=key))
        with patch.object(ui, "Screen", SelectWatch):
            self.assertEqual(ui.menu(), ["watch"])

    def test_space_reaches_human_physics(self):
        original = ui.Screen
        environments = []
        class PressSpace(original):
            def __init__(self):
                super().__init__()
                ui.pg.event.post(ui.pg.event.Event(ui.pg.KEYDOWN, key=ui.pg.K_SPACE))
        def tracked_runner(config):
            env = Runner(config)
            environments.append(env)
            return env
        args = SimpleNamespace(command="play", endless=False, seed=42, frames=8, screenshot=None)
        with patch.object(ui, "Screen", PressSpace), patch.object(ui, "Runner", tracked_runner):
            ui.play(args)
        self.assertGreater(environments[0].y, 0)
        self.assertEqual(environments[0].ticks, 8)

    def test_training_pause_speed_and_exit(self):
        view = ui.TrainingView()
        timer = None
        try:
            ui.pg.event.post(ui.pg.event.Event(ui.pg.KEYDOWN, key=ui.pg.K_TAB))
            ui.pg.event.post(ui.pg.event.Event(ui.pg.KEYDOWN, key=ui.pg.K_p))
            timer = threading.Timer(.15, lambda: ui.pg.event.post(
                ui.pg.event.Event(ui.pg.KEYDOWN, key=ui.pg.K_p)))
            timer.start()
            self.assertTrue(view.update(Runner(), QNetwork(), 1, 1, 1, None, [], 0, True))
            self.assertTrue(view.fast)
            self.assertFalse(view.paused)
            ui.pg.event.post(ui.pg.event.Event(ui.pg.KEYDOWN, key=ui.pg.K_ESCAPE))
            self.assertFalse(view.update(Runner(), QNetwork(), 1, 1, 1, None, [], 0, True))
        finally:
            if timer:
                timer.join()
            view.close()


if __name__ == "__main__":
    unittest.main()
