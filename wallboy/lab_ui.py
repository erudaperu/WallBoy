"""Small Pygame experiment form and saved-run comparison screen."""
from datetime import datetime
import io
import json
from pathlib import Path

from .ui import Screen, pg, CYAN, PINK, WHITE, MUTED, GOLD, PANEL
from .settings import OPTIONS, GROUPS, DEFAULTS, Option, validate
from .history import METRICS, discover_runs, comparison_png

ROOT = Path(__file__).resolve().parents[1] / 'runs'


def button(screen, label, rect, active=False):
    pg.draw.rect(screen.surface, (30, 63, 69) if active else PANEL, rect, border_radius=6)
    screen.text(label, rect.x+10, rect.y+8, CYAN if active else WHITE, screen.small,
                max_width=rect.w-15)


def training_settings():
    screen = Screen()
    path = ROOT/'training-preferences.json'
    try:
        saved = json.loads(path.read_text())
        values = {k:str(v) for k,v in validate(saved).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        values = {k:str(v) for k,v in DEFAULTS.items()}
    group, focus, info, error = 0, None, OPTIONS[0], ''
    device, live = 'cpu', True
    tabs = [pg.Rect(40+i*210, 120, 200, 42) for i in range(5)]
    start = pg.Rect(40, 660, 270, 55)
    reset = pg.Rect(330, 660, 180, 55)
    device_rect = pg.Rect(40, 575, 300, 50)
    live_rect = pg.Rect(400, 575, 300, 50)
    extra_help = [
        (pg.Rect(350,580,38,38), Option('device','Device','cpu',0,0,
         'CPU works with the included installation. Auto uses CUDA when your selected Python environment supports it. CUDA explicitly requests the NVIDIA GPU. This small network is often fast enough on CPU.', 'Run')),
        (pg.Rect(710,580,38,38), Option('live','Watch live','yes',0,0,
         'YES opens the game and shows decisions during training. Tab switches between slow live learning and fast training. NO runs in the terminal. Both save the same CSV data for your graph comparison.', 'Run'))]
    try:
        while True:
            visible = [o for o in OPTIONS if o.group == GROUPS[group]]
            fields = [(o, pg.Rect(335, 195+i*78, 200, 49), pg.Rect(555, 200+i*78, 38, 38))
                      for i,o in enumerate(visible)]
            for event in screen.events():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    return None
                if event.type == pg.MOUSEMOTION:
                    pos = screen.logical_position(event.pos)
                    for option, _, icon in fields:
                        if icon.collidepoint(pos):
                            info = option
                    for icon, option in extra_help:
                        if icon.collidepoint(pos):
                            info = option
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = screen.logical_position(event.pos)
                    for i, rect in enumerate(tabs):
                        if rect.collidepoint(pos):
                            group, focus = i, None
                            info = next(o for o in OPTIONS if o.group == GROUPS[i])
                    focus = next((o.key for o,rect,_ in fields if rect.collidepoint(pos)), None)
                    if focus:
                        info = next(o for o in OPTIONS if o.key == focus)
                    for option, _, icon in fields:
                        if icon.collidepoint(pos):
                            info = option
                    for icon, option in extra_help:
                        if icon.collidepoint(pos):
                            info = option
                    if reset.collidepoint(pos):
                        values = {k:str(v) for k,v in DEFAULTS.items()}
                        error = ''
                    if device_rect.collidepoint(pos):
                        device = ['cpu', 'auto', 'cuda'][(['cpu','auto','cuda'].index(device)+1)%3]
                    if live_rect.collidepoint(pos):
                        live = not live
                    if start.collidepoint(pos):
                        try:
                            settings = validate(values)
                            from .agent import choose_device
                            choose_device(device)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps(settings, indent=2))
                            output = ROOT/datetime.now().strftime('experiment-%Y%m%d-%H%M%S-%f')
                            args = ['train', '--out', str(output), '--device', device]
                            for key,value in settings.items():
                                args.extend(['--'+key.replace('_','-'), str(value)])
                            return args + (['--live'] if live else [])
                        except (ValueError, OSError) as exc:
                            error = str(exc)
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_TAB:
                        keys = [o.key for o in visible]
                        focus = keys[(keys.index(focus)+1)%len(keys)] if focus in keys else keys[0]
                        info = next(o for o in OPTIONS if o.key == focus)
                    elif focus:
                        if event.key == pg.K_BACKSPACE:
                            values[focus] = values[focus][:-1]
                        elif event.key == pg.K_a and getattr(event, 'mod', 0) & pg.KMOD_CTRL:
                            values[focus] = ''
                        elif getattr(event, 'unicode', '') and event.unicode in '0123456789.eE-+':
                            if len(values[focus]) < 14:
                                values[focus] += event.unicode
            visible = [o for o in OPTIONS if o.group == GROUPS[group]]
            fields = [(o, pg.Rect(335, 195+i*78, 200, 49), pg.Rect(555, 200+i*78, 38, 38))
                      for i,o in enumerate(visible)]
            screen.header('training / experiment lab')
            for i,rect in enumerate(tabs):
                button(screen, GROUPS[i], rect, i==group)
            for option, rect, icon in fields:
                screen.text(option.label, 50, rect.y+10, WHITE, screen.small, max_width=280)
                button(screen, values[option.key]+('|' if focus==option.key else ''), rect, focus==option.key)
                pg.draw.circle(screen.surface, CYAN, icon.center, 16, 2)
                screen.text('i', icon.centerx-screen.small.size('i')[0]/2, icon.y+4, CYAN, screen.small)
            pg.draw.rect(screen.surface, PANEL, (620, 185, 460, 365), border_radius=10)
            bottom = screen.text(info.label, 642, 205, CYAN, max_width=410)
            bottom = screen.text(info.help, 642, bottom+18, WHITE, screen.small, max_width=410)
            screen.text(f'Default {info.default}' + (f' | Range {info.minimum:g} to {info.maximum:g}' if info.key not in ('device','live') else ''),
                        642, bottom+16, GOLD, screen.small, max_width=410)
            button(screen, 'Device: '+device.upper(), device_rect)
            button(screen, 'Watch live: '+('YES' if live else 'NO'), live_rect)
            for icon, _ in extra_help:
                pg.draw.circle(screen.surface, CYAN, icon.center, 16, 2)
                screen.text('i', icon.centerx-screen.small.size('i')[0]/2, icon.y+4, CYAN, screen.small)
            screen.text('Hover i to learn.', 780, 587, MUTED, screen.small)
            screen.text(error or 'Click a value | Ctrl+A clears | Hover i for help', 40, 632,
                        PINK if error else MUTED, screen.small)
            button(screen, 'Start training', start, True)
            button(screen, 'Defaults', reset)
            screen.text('Fresh Classic agent. ESC home.', 535, 675, MUTED, screen.small)
            screen.present()
            screen.clock.tick(30)
    finally:
        screen.close()


def graph_gallery():
    screen = Screen()
    runs, skipped = discover_runs(ROOT)
    selected = {r['name'] for r in runs}
    metric, axis, smoothing, page = 'Reward', 'episode', 20, 0
    chart, old_key, note = None, None, ''
    metrics = list(METRICS)
    tabs = [pg.Rect(40+i*175, 120, 165, 40) for i in range(6)]
    buttons = [(pg.Rect(40, 700, 140, 36), 'Refresh'), (pg.Rect(190,700,130,36), 'All/none'),
               (pg.Rect(330,700,160,36), 'Smooth'), (pg.Rect(500,700,170,36), 'X axis'),
               (pg.Rect(680,700,170,36), 'Export PNG'), (pg.Rect(860,700,100,36), 'Prev'),
               (pg.Rect(970,700,110,36), 'Next')]
    try:
        while True:
            shown = runs[page*6:page*6+6]
            rows = [(run, pg.Rect(40+(i%2)*530, 550+(i//2)*43, 510, 39)) for i,run in enumerate(shown)]
            for event in screen.events():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    return
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = screen.logical_position(event.pos)
                    for i, rect in enumerate(tabs):
                        if rect.collidepoint(pos):
                            metric = metrics[i]
                    for run, rect in rows:
                        if rect.collidepoint(pos):
                            if run['name'] in selected:
                                selected.remove(run['name'])
                            else:
                                selected.add(run['name'])
                            args = run['settings'].get('arguments', {})
                            note = f"{run['name']} | lr {args.get('learning_rate', run['settings'].get('learning_rate', '?'))} | seed {args.get('seed', '?')}"
                    for rect,label in buttons:
                        if not rect.collidepoint(pos):
                            continue
                        if label == 'Refresh':
                            old_names = {r['name'] for r in runs}
                            runs, skipped = discover_runs(ROOT)
                            selected |= {r['name'] for r in runs}-old_names
                            old_key = None
                        elif label == 'All/none':
                            selected = set() if selected else {r['name'] for r in runs}
                        elif label == 'Smooth':
                            smoothing = 1 if smoothing == 20 else 20
                        elif label == 'X axis':
                            axis = 'steps' if axis == 'episode' else 'episode'
                        elif label == 'Prev':
                            page = max(0, page-1)
                        elif label == 'Next':
                            page = min(max(0,(len(runs)-1)//6), page+1)
                        elif label == 'Export PNG':
                            chosen = [r for r in runs if r['name'] in selected]
                            ROOT.mkdir(parents=True, exist_ok=True)
                            destination = ROOT/('comparison-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.png')
                            destination.write_bytes(comparison_png(chosen, METRICS[metric], axis, smoothing, screen.text_scale))
                            note = 'Saved '+destination.name
            chosen = [r for r in runs if r['name'] in selected]
            key = (tuple(sorted(selected)), metric, axis, smoothing, screen.text_scale)
            if key != old_key:
                chart = pg.image.load(io.BytesIO(comparison_png(chosen, METRICS[metric], axis, smoothing, screen.text_scale))).convert()
                old_key = key
            screen.header('graphs / your experiments')
            for i,rect in enumerate(tabs):
                button(screen, metrics[i], rect, metric==metrics[i])
            screen.surface.blit(pg.transform.smoothscale(chart, (1040,365)), (40,165))
            for run,rect in rows:
                name = run['name']
                label = ('[x] ' if name in selected else '[ ] ')+name
                # Keep full name in chart legend; shorten only selection buttons.
                while screen.small.size(label)[0] > rect.w-25:
                    label = label[:-4]+'...'
                button(screen, label, rect, name in selected)
            reward_rules = {tuple(r['settings'].get('environment', {}).get(k,d) for k,d in
                                 [('alive_reward',.02),('clear_reward',3),('jump_cost',.03),('death_cost',5),('max_ticks',1800)]) for r in chosen}
            message = 'Different rewards/time limits: compare Cleared and Survival too.' if len(reward_rules)>1 else note or f'{len(chosen)}/{len(runs)} runs | Average: {smoothing} episodes | Page {page+1} | ESC home'
            if skipped:
                message = f'{len(skipped)} unreadable files skipped. '+message
            screen.text(message, 40, 679, GOLD if len(reward_rules)>1 else MUTED, screen.small)
            for rect,label in buttons:
                button(screen, label, rect)
            screen.present()
            screen.clock.tick(30)
    finally:
        screen.close()
