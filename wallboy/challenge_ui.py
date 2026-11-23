"""Human controls for the harder game. No automated player, no training."""
import json
from pathlib import Path

from .challenge import Challenge
from .ui import Screen, pg, CYAN, PINK, GOLD, MUTED, WHITE


def play_challenge(args):
    screen = Screen()
    nightmare = args.nightmare
    seed = args.seed
    env = Challenge(seed, nightmare)
    record_path = Path("runs/arcade-record.json")
    try:
        records = json.loads(record_path.read_text())
        if not isinstance(records, dict):
            records = {}
    except (OSError, ValueError):
        records = {}
    def best_score():
        value = records.get("nightmare" if nightmare else "hard", 0)
        return value if isinstance(value, (int, float)) else 0
    def save_record():
        if env.score > best_score():
            records["nightmare" if nightmare else "hard"] = env.score
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(records, indent=2))
    frames = 0
    paused = False
    running = True
    # Fixed-step simulation keeps speed fair even if drawing drops a frame.
    accumulator = 0.0
    pending = []
    try:
        while running:
            elapsed = min(screen.clock.tick(60) / 1000, .1)
            for event in screen.events():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN and not getattr(event, "repeat", False):
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key == pg.K_p:
                        paused = not paused
                        pending.clear()
                        accumulator = 0
                    elif event.key in (pg.K_r, pg.K_n, pg.K_h):
                        save_record()
                        if event.key == pg.K_n:
                            seed += 1
                        if event.key == pg.K_h:
                            nightmare = not nightmare
                        env = Challenge(seed, nightmare)
                        paused = False
                        pending.clear()
                        accumulator = 0
                    elif not paused and not env.dead:
                        if event.key in (pg.K_SPACE, pg.K_UP):
                            pending.append(1)
                        elif event.key in (pg.K_DOWN, pg.K_s):
                            pending.append(2)
            if running and not paused and not env.dead:
                accumulator += elapsed
                while accumulator >= 1/60 and not env.dead:
                    env.tick(pending.pop(0) if pending else 0)
                    accumulator -= 1/60
                if env.dead:
                    save_record()
            screen.header("nightmare" if nightmare else "hard / arcade")
            screen.text("DOUBLE JUMP. BREAK YOUR RECORD.", 58, 127, WHITE)
            screen.lane(env, 168, "PAUSED" if paused else "YOU / TWO JUMPS", height=324)
            screen.text(f"SCORE {env.score:06d}   x{env.multiplier}   BEST {int(max(best_score(), env.score)):06d}",
                        58, 518, GOLD)
            screen.text(f"JUMPS {'O ' * env.jumps_left}{'- ' * (2-env.jumps_left)}   "
                        f"SPEED {env.speed:.1f}   LEVEL {env.level}   ORBS {env.collected}", 58, 552, CYAN)
            notice = "CRASHED / R retry / N new course" if env.dead else env.notice if env.ticks < env.notice_until else ""
            if not env.dead and env.ticks < env.level_up_until:
                notice = f"LEVEL {env.level}! / SPEED {env.speed:.1f}"
            screen.text(notice, 58, 590, PINK)
            screen.text("Towers: double jump | Lasers: stay low | Orbs: bonus points", 58, 630, MUTED, screen.small)
            screen.text("SPACE/UP double jump   DOWN/S drop   H difficulty", 58, 675, CYAN, screen.small)
            screen.text("P pause    R retry    N new course    ESC exit", 58, 708, MUTED, screen.small)
            screen.present()
            frames += 1
            if args.frames and frames >= args.frames:
                running = False
        if args.screenshot:
            screen.save(args.screenshot)
    finally:
        save_record()
        screen.close()
