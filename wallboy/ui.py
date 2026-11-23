"""Pygame presentation only. The game and agent also work without a display."""
from dataclasses import replace
from datetime import datetime
import os
import json
from pathlib import Path
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame as pg
import torch

from .agent import greedy_action, load_policy, q_values
from .env import Config, Runner

BG = (9, 11, 24)
PANEL = (17, 22, 41)
CYAN = (83, 244, 219)
PINK = (255, 99, 199)
WHITE = (228, 234, 248)
MUTED = (142, 155, 181)
GOLD = (255, 205, 112)


class Screen:
    SIZE = (1120, 800)
    SETTINGS = Path(__file__).resolve().parents[1] / "runs/display-settings.json"

    def __init__(self, settings_path=None):
        pg.display.init()
        pg.font.init()
        # Headless tests never overwrite your desktop preferences.
        self.settings_path = Path(settings_path) if settings_path else (
            self.SETTINGS if pg.display.get_driver() != "dummy" else None)
        settings = {}
        try:
            settings = json.loads(self.settings_path.read_text()) if self.settings_path else {}
            if not isinstance(settings, dict):
                settings = {}
        except (OSError, ValueError):
            pass
        desktop = pg.display.get_desktop_sizes()[0]
        default = (min(1600, int(desktop[0]*.85)), min(1143, int(desktop[1]*.85)))
        size = settings.get("window_size", default)
        if not (isinstance(size, (list, tuple)) and len(size) == 2
                and all(type(n) is int and n >= 320 for n in size)):
            size = default
        size = (min(size[0], desktop[0]), min(size[1], desktop[1]))
        self.window = pg.display.set_mode(size, pg.RESIZABLE)
        self.surface = pg.Surface(self.SIZE)
        pg.display.set_caption("WALLBOY / AI RIVAL")
        self.set_text_scale(settings.get("text_scale", 1.25))
        self.clock = pg.time.Clock()
        self.text_buttons = [(pg.Rect(45, 751, 55, 38), -0.1, "-"),
                             (pg.Rect(255, 751, 55, 38), 0.1, "+"),
                             (pg.Rect(325, 751, 115, 38), 0, "Reset")]

    def set_text_scale(self, value):
        if not isinstance(value, (int, float)) or not .8 <= value <= 1.75:
            value = 1.25
        self.text_scale = round(value, 2)
        self.font = pg.font.SysFont("DejaVu Sans Mono", round(16*value))
        self.small = pg.font.SysFont("DejaVu Sans Mono", round(13*value))
        self.large = pg.font.SysFont("DejaVu Sans Mono", round(34*(1+.35*(value-1))), bold=True)

    def change_text(self, delta):
        self.set_text_scale(min(1.75, max(.8, self.text_scale+delta)) if delta else 1.25)

    def viewport(self):
        width, height = self.window.get_size()
        ratio = min(width/self.SIZE[0], height/self.SIZE[1])
        size = (max(1, round(self.SIZE[0]*ratio)), max(1, round(self.SIZE[1]*ratio)))
        return pg.Rect((width-size[0])//2, (height-size[1])//2, *size)

    def logical_position(self, pos):
        rect = self.viewport()
        return ((pos[0]-rect.x)*self.SIZE[0]/rect.w,
                (pos[1]-rect.y)*self.SIZE[1]/rect.h)

    def events(self):
        for event in pg.event.get():
            if event.type == pg.VIDEORESIZE:
                self.window = pg.display.set_mode((max(320, event.w), max(240, event.h)), pg.RESIZABLE)
                continue
            if event.type == pg.KEYDOWN and getattr(event, "mod", 0) & pg.KMOD_CTRL:
                if event.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS, pg.K_MINUS, pg.K_KP_MINUS, pg.K_0):
                    self.change_text(0 if event.key == pg.K_0 else -.1 if event.key in
                                     (pg.K_MINUS, pg.K_KP_MINUS) else .1)
                    continue
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos = self.logical_position(event.pos)
                button = next((b for b in self.text_buttons if b[0].collidepoint(pos)), None)
                if button:
                    self.change_text(button[1])
                    continue
            yield event

    def text(self, text, x, y, color=WHITE, font=None, max_width=None):
        font = font or self.font
        width = max_width if max_width is not None else 1080-x
        lines = []
        for paragraph in str(text).splitlines() or [""]:
            line = ""
            for word in paragraph.split():
                candidate = (line+" "+word).strip()
                if line and font.size(candidate)[0] > width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        for line in lines:
            self.surface.blit(font.render(line, True, color), (x, y))
            y += font.get_linesize()
        return y

    def present(self):
        pg.draw.rect(self.surface, PANEL, (0, 742, 1120, 58))
        for rect, _, label in self.text_buttons:
            pg.draw.rect(self.surface, (35, 57, 70), rect, border_radius=5)
            self.text(label, rect.centerx-self.small.size(label)[0]/2,
                      rect.centery-self.small.get_linesize()/2, CYAN, self.small)
        self.text(f"Text {self.text_scale:.0%}", 113, 760, WHITE, self.small)
        self.text("Resize edges | Ctrl +/- | Ctrl 0", 470, 760, MUTED, self.small)
        self.window.fill(BG)
        rect = self.viewport()
        self.window.blit(pg.transform.smoothscale(self.surface, rect.size), rect)
        pg.display.flip()

    def header(self, mode):
        self.surface.fill(BG)
        self.text("WALLBOY", 40, 26, CYAN, self.large)
        self.text("/ AI RIVAL", 60+self.large.size("WALLBOY")[0], 42, WHITE)
        self.text("CLOCK  /  TIMER  /  ALARM  /  ARCADE", 40, 76, MUTED, self.small)
        self.text(datetime.now().strftime("%H:%M:%S"), 1080-self.font.size("00:00:00")[0], 30, GOLD)
        self.text(mode.upper(), 1080-self.small.size(mode.upper())[0], 76, PINK, self.small)
        pg.draw.line(self.surface, (39, 47, 76), (40, 108), (1080, 108))

    def lane(self, env, top, label, color=CYAN, height=230):
        left, width = 40, 1040
        floor = top + height - 32
        pg.draw.rect(self.surface, PANEL, (left, top, width, height), border_radius=12)
        previous_clip = self.surface.get_clip()
        self.surface.set_clip((left, top, width, height))
        # Deterministic, decorative skyline. It never informs the agent.
        for i in range(22):
            x = left + (i * 58 - env.ticks * env.speed * .14) % width
            tower_height = 22 + (i * 29) % 64
            pg.draw.rect(self.surface, (24, 29, 51), (x, floor - tower_height, 35, tower_height))
            pg.draw.line(self.surface, (43, 41, 72), (x+8, floor-tower_height+12),
                         (x+23, floor-tower_height+12), 2)
        pg.draw.line(self.surface, color, (left, floor), (left+width, floor), 2)
        for x in range(0, width, 36):
            offset = int(env.ticks * env.speed) % 36
            pg.draw.line(self.surface, (35, 47, 67), (left+x-offset, floor+10),
                         (left+x-offset+16, floor+10))
        scale = width / env.config.width
        for obstacle in env.obstacles:
            x = left + obstacle.x * scale
            if left <= x < left+width-48:
                bottom = getattr(obstacle, "bottom", 0)
                rect = pg.Rect(x, floor-bottom-obstacle.height, obstacle.width*scale, obstacle.height)
                pg.draw.rect(self.surface, (87, 33, 80), rect, border_radius=3)
                tint = GOLD if bottom else PINK
                pg.draw.rect(self.surface, tint, rect, width=2, border_radius=3)
                pg.draw.line(self.surface, tint, rect.topleft, rect.bottomright, 1)
                if getattr(obstacle, "kind", "") == "laser":
                    self.text("STAY LOW", x, floor-bottom-38, GOLD, self.small)
        for orb in getattr(env, "orbs", []):
            pos = (int(left + orb.x * scale), int(floor - orb.y))
            pg.draw.circle(self.surface, (36, 78, 76), pos, 12)
            pg.draw.circle(self.surface, CYAN, pos, 6, 2)
        c = env.config
        player = pg.Rect(left+c.player_x*scale, floor-env.y-c.player_height,
                         c.player_width*scale, c.player_height)
        pg.draw.rect(self.surface, (77, 48, 63) if env.dead else color, player, border_radius=5)
        pg.draw.rect(self.surface, BG, (player.x+7, player.y+9, 18, 9), border_radius=2)
        pg.draw.rect(self.surface, WHITE, (player.x+17, player.y+11, 4, 4))
        self.text(label, left+18, top+14, color)
        status = f"{env.ticks/60:05.1f}s   CLEARED {env.cleared:02d}"
        self.text(status, 1062-self.font.size(status)[0], top+14, WHITE)
        if env.done:
            self.text("SIGNAL LOST" if env.dead else "30s ROUND COMPLETE" if c.max_ticks == 1800
                      else "ROUND COMPLETE", left+400, floor-80, PINK if env.dead else CYAN)
        self.surface.set_clip(previous_clip)

    def graph(self, rewards, x=58, y=480, width=620, height=165):
        self.text("EPISODE REWARDS / last 200", x, y-25, MUTED, self.small)
        pg.draw.rect(self.surface, PANEL, (x, y, width, height), border_radius=6)
        values = rewards[-200:]
        if len(values) < 2:
            self.text("Collecting first episodes...", x+20, y+65, MUTED, self.small)
            return
        lo, hi = min(-6, min(values)), max(10, max(values))
        def points(series):
            return [(x+i*(width-1)/(len(series)-1), y+height-8-(v-lo)/(hi-lo)*(height-16))
                    for i,v in enumerate(series)]
        pg.draw.lines(self.surface, (62, 79, 110), False, points(values), 1)
        smooth = [sum(values[max(0,i-19):i+1])/len(values[max(0,i-19):i+1])
                  for i in range(len(values))]
        pg.draw.lines(self.surface, CYAN, False, points(smooth), 2)
        self.text(f"{hi:.0f}", x+5, y+5, MUTED, self.small)
        self.text(f"{lo:.0f}", x+5, y+height-self.small.get_linesize()-4, MUTED, self.small)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.present()
        pg.image.save(self.window, path)

    def close(self):
        if self.settings_path:
            try:
                self.settings_path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.settings_path.with_suffix(".tmp")
                temp.write_text(json.dumps({"window_size": self.window.get_size(),
                                            "text_scale": self.text_scale}, indent=2))
                temp.replace(self.settings_path)
            except OSError:
                pass  # A read-only folder must not prevent closing the game.
        pg.quit()


class TrainingView(Screen):
    def __init__(self):
        super().__init__()
        self.fast = self.paused = False
        self.last_draw = 0

    def update(self, env, model, episode, steps, epsilon, loss, rewards, action, exploratory):
        while True:
            for event in self.events():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    return False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_TAB:
                        self.fast = not self.fast
                    if event.key == pg.K_p:
                        self.paused = not self.paused
            now = time.monotonic()
            if not self.fast or self.paused or now - self.last_draw >= 1/30:
                self.last_draw = now
                self.header("training / " + ("paused" if self.paused else "fast" if self.fast else "live"))
                self.lane(env, 130, "LEARNING AGENT")
                self.text(f"EPISODE {episode:04d}    DECISIONS {steps:07d}", 58, 384)
                self.text(f"EXPLORATION {epsilon:.0%}    LAST ACTION {'JUMP' if action else 'WAIT'}"
                          f" / {'random' if exploratory else 'network'}", 58, 415, GOLD, self.small)
                self.graph(rewards)
                values = q_values(model, env.observe())
                self.text("NEXT DECISION / Q", 740, 480, PINK)
                self.text(f"WAIT  {values[0]:8.3f}", 740, 520)
                self.text(f"JUMP  {values[1]:8.3f}", 740, 550)
                self.text("LOSS  " + (f"{loss:.4f}" if loss is not None else "warming up"), 740, 595, MUTED)
                self.text("Learns from experience.", 740, 630, MUTED, self.small)
                self.text("TAB speed   P pause   ESC save + finish", 58, 704, CYAN)
                self.present()
            if self.paused:
                self.clock.tick(30)
                continue
            if not self.fast:
                self.clock.tick(15)  # Four 60 Hz physics ticks per decision.
            return True


def play(args):
    torch.set_num_threads(1)
    model = None
    config = Config()
    if args.command != "play":
        model, config, _ = load_policy(args.model)
    if args.endless:
        config = replace(config, max_ticks=0)
    screen = Screen()
    seed = args.seed
    human, bot = Runner(config), Runner(config)
    def reset():
        human.reset(seed)
        bot.reset(seed)
    reset()
    paused = pending_jump = False
    frames = 0
    running = True
    try:
        while running:
            for event in screen.events():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key in (pg.K_SPACE, pg.K_UP):
                        pending_jump = True
                    elif event.key == pg.K_p:
                        paused = not paused
                    elif event.key in (pg.K_r, pg.K_n):
                        if event.key == pg.K_n:
                            seed += 1
                        reset()
                        pending_jump = False
            if not paused and running:
                if args.command != "watch" and not human.done:
                    action = 0
                    if human.ticks % config.action_ticks == 0:
                        action, pending_jump = int(pending_jump), False
                    human.tick(action)
                if model is not None and not bot.done:
                    action = greedy_action(model, bot.observe()) if bot.ticks % config.action_ticks == 0 else 0
                    bot.tick(action)
            screen.header(args.command + (" / paused" if paused else " / endless" if args.endless
                                          else f" / {config.max_ticks / 60:g}s sprint"))
            if args.command == "race":
                screen.lane(human, 130, "YOU / SPACE TO JUMP")
                screen.lane(bot, 382, "TRAINED RIVAL", PINK)
                if human.done and bot.done:
                    human_score, bot_score = (human.cleared, human.ticks), (bot.cleared, bot.ticks)
                    result = "YOU WIN" if human_score > bot_score else "RIVAL WINS" if bot_score > human_score else "DRAW"
                    screen.text(result + " / R replay / N new track", 58, 637, GOLD)
                else:
                    screen.text(f"SAME COURSE / SEED {seed}    Score: clears, then survival time", 58, 637, MUTED, screen.small)
            else:
                env = human if args.command == "play" else bot
                screen.lane(env, 160, "YOU / SPACE TO JUMP" if model is None else "TRAINED RIVAL", height=300)
                screen.text(f"COURSE {seed}    REWARD {env.total_reward:.2f}", 58, 490, GOLD)
                if model is not None:
                    q = q_values(model, bot.observe())
                    screen.text(f"Q(WAIT) {q[0]:.3f}    Q(JUMP) {q[1]:.3f}", 58, 530, MUTED)
                screen.text("One button. Learn the rhythm. Beat your own creation.", 58, 606, CYAN)
            screen.text("SPACE jump   P pause   R replay   N new course   ESC exit", 58, 704, CYAN)
            screen.present()
            frames += 1
            if args.frames and frames >= args.frames:
                running = False
            screen.clock.tick(60)
        if args.screenshot:
            screen.save(args.screenshot)
    finally:
        screen.close()


def menu():
    screen = Screen()
    options = ["PLAY / double-jump challenge", "WATCH / classic trained rival",
               "RACE / classic AI race", "TRAIN / settings + explanations", "GRAPHS / compare your experiments"]
    selected = 0
    def launch(index):
        return [["play"], ["watch"], ["race"], ["settings"], ["graphs"]][index]
    try:
        while True:
            for event in screen.events():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    return None
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_DOWN:
                        selected = (selected+1) % len(options)
                    elif event.key == pg.K_UP:
                        selected = (selected-1) % len(options)
                    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        return launch(selected)
                if event.type in (pg.MOUSEMOTION, pg.MOUSEBUTTONDOWN):
                    pos = screen.logical_position(event.pos)
                    for i in range(len(options)):
                        if pg.Rect(50, 285+i*62, 1020, 54).collidepoint(pos):
                            selected = i
                            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                                return launch(i)
            screen.header("arcade / home")
            screen.text("BUILD A RIVAL. THEN BEAT IT.", 58, 155, WHITE, screen.large)
            screen.text("A tiny neural network inside your Wallboy universe.", 58, 220, MUTED)
            for i, label in enumerate(options):
                y = 285+i*62
                pg.draw.rect(screen.surface, (27, 43, 58) if i == selected else PANEL,
                             (50, y, 1020, 54), border_radius=8)
                screen.text(("> " if i == selected else "  ")+label, 72, y+12,
                            CYAN if i == selected else MUTED)
            screen.text("UP / DOWN select    ENTER launch    ESC exit", 58, 646, CYAN)
            screen.text("TRAIN: Tab = fast/live. New runs start from scratch.", 58, 704, MUTED, screen.small)
            screen.present()
            screen.clock.tick(30)
    finally:
        screen.close()
