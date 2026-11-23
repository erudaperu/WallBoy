"""Real display controls, parameter wiring and saved experiment comparison."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from wallboy import ui, lab_ui
from wallboy.agent import DQN, load_policy, save_policy
from wallboy.env import Runner, Config
from wallboy.history import discover_runs, series, comparison_png
from wallboy.settings import validate, exploration

class LabTests(unittest.TestCase):
    def test_display_preferences_and_coordinate_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'display.json'
            s=ui.Screen(path)
            ui.pg.event.clear()
            ui.pg.event.post(ui.pg.event.Event(ui.pg.VIDEORESIZE,w=1500,h=900))
            ui.pg.event.post(ui.pg.event.Event(ui.pg.KEYDOWN,key=ui.pg.K_EQUALS,mod=ui.pg.KMOD_CTRL))
            list(s.events())
            self.assertEqual(s.window.get_size(),(1500,900))
            self.assertEqual(s.text_scale,1.35)
            r=s.viewport()
            pos=(r.x+280*r.w/1120,r.y+770*r.h/800)
            ui.pg.event.post(ui.pg.event.Event(ui.pg.MOUSEBUTTONDOWN,button=1,pos=pos))
            list(s.events())
            self.assertEqual(s.text_scale,1.45)
            s.header('test'); s.present(); s.close()
            self.assertEqual(json.loads(path.read_text())['window_size'],[1500,900])
            s=ui.Screen(path)
            self.assertEqual(s.text_scale,1.45)
            s.close()
            path.write_text('{bad')
            s=ui.Screen(path)
            self.assertEqual(s.text_scale,1.25)
            s.close()

    def test_learning_parameters_and_variable_network_checkpoint(self):
        torch.set_num_threads(1)
        a=DQN(learning_rate=.001,hidden_size=16,gamma=.5,batch_size=2,replay_size=8,warmup_steps=2,target_update=1)
        self.assertEqual(a.online.layers[0].out_features,16)
        self.assertEqual(a.optimizer.param_groups[0]['lr'],.001)
        self.assertEqual(a.memory.maxlen,8)
        self.assertIsNone(a.learn())
        with torch.no_grad():
            for net in (a.online,a.target):
                for p in net.parameters(): p.zero_()
                net.layers[-1].bias.fill_(2)
        obs=Runner().observe()
        a.memory.extend([(obs,0,0.,obs,False)]*2)
        self.assertAlmostEqual(a.learn(),.5,places=5)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.online.parameters(),a.target.parameters())))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'tiny.pt'
            save_policy(p,a.online,Runner().metadata())
            self.assertEqual(load_policy(p)[0].hidden_size,16)

    def test_validation_exploration_and_rewards(self):
        for values in ({'batch_size':100,'replay_size':10},{'warmup_steps':500,'replay_size':10},
                       {'epsilon_start':.1,'epsilon_end':.5},{'gamma':'nan'},{'episodes':2.5}):
            with self.assertRaises(ValueError): validate(values)
        values=validate({'epsilon_start':.8,'epsilon_end':.2,'explore_steps':100})
        self.assertAlmostEqual(exploration(50,values),.5)
        self.assertAlmostEqual(exploration(200,values),.2)
        self.assertAlmostEqual(Runner(Config(alive_reward=.4,jump_cost=.2)).tick(1),.2)

    def test_overlay_uses_real_coordinates_and_skips_missing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            for name,rewards in [('first',[1,3]),('second',[5,9])]:
                p=Path(folder)/name; p.mkdir()
                with (p/'episodes.csv').open('w') as f:
                    w=csv.writer(f); w.writerow(['episode','steps','reward','validation_reward'])
                    for i,r in enumerate(rewards,1): w.writerow([i,i*10,r,7 if i==2 else ''])
            runs, skipped=discover_runs(folder)
            self.assertEqual(len(runs),2); self.assertFalse(skipped)
            self.assertEqual(series(runs[0],'reward','steps',2),([10.,20.],[1.,2.]))
            self.assertEqual(series(runs[0],'validation_reward'),([2.],[7.]))
            self.assertTrue(comparison_png(runs,'reward').startswith(b'\x89PNG'))

    def test_form_edits_reach_training_arguments(self):
        class Scripted(ui.Screen):
            def events(self):
                r=self.viewport()
                def click(x,y):
                    return ui.pg.event.Event(ui.pg.MOUSEBUTTONDOWN,button=1,pos=(r.x+x*r.w/1120,r.y+y*r.h/800))
                yield click(400,215)
                yield ui.pg.event.Event(ui.pg.KEYDOWN,key=ui.pg.K_a,mod=ui.pg.KMOD_CTRL)
                for ch in '12': yield ui.pg.event.Event(ui.pg.KEYDOWN,key=ord(ch),unicode=ch)
                yield click(100,680)
        with tempfile.TemporaryDirectory() as folder,patch.object(lab_ui,'ROOT',Path(folder)),patch.object(lab_ui,'Screen',Scripted):
            args=lab_ui.training_settings()
            self.assertEqual(args[args.index('--episodes')+1],'12')
            self.assertIn('--learning-rate',args)
            self.assertIn('--live',args)
