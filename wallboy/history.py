"""Read saved experiments and draw one colored line per selected run."""
import csv
import io
import json
import math
import os
from pathlib import Path

METRICS = {'Reward': 'reward', 'Validation': 'validation_reward', 'Cleared': 'cleared',
           'Survival': 'seconds', 'Exploration': 'epsilon', 'Loss': 'loss'}
COLORS = ['#53f4db', '#ff63c7', '#ffcd70', '#7aaaff', '#b79cff', '#ff906a', '#8ce981', '#ffffff']


def discover_runs(root):
    runs, skipped = [], []
    for path in sorted(Path(root).rglob('episodes.csv'), key=lambda p:p.stat().st_mtime):
        try:
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            valid = []
            for row in rows:
                try:
                    clean = {k:float(v) for k,v in row.items() if k and v and k in
                             ('episode', 'steps', *METRICS.values())}
                    if 'episode' not in clean or not all(math.isfinite(v) for v in clean.values()):
                        continue
                    valid.append(clean)
                except (ValueError, TypeError):
                    continue  # An incomplete final CSV line may still be being written.
            settings_path = path.parent/'settings.json'
            try:
                settings = json.loads(settings_path.read_text())
                if not isinstance(settings, dict):
                    settings = {}
            except (OSError, ValueError):
                settings = {}
            runs.append({'name':str(path.parent.relative_to(root)), 'path':path.parent,
                         'rows':valid, 'settings':settings})
        except (OSError, csv.Error, UnicodeError):
            skipped.append(str(path))
    return runs, skipped


def series(run, metric, axis='episode', smoothing=1):
    points = [(row[axis], row[metric]) for row in run['rows'] if axis in row and metric in row]
    if not points:
        return [], []
    xs, ys = zip(*points)
    averaged = [sum(ys[max(0,i-smoothing+1):i+1])/len(ys[max(0,i-smoothing+1):i+1])
                for i in range(len(ys))]
    return list(xs), averaged


def comparison_png(runs, metric, axis='episode', smoothing=1, text_scale=1.25):
    # Agg renders to memory; importing this module never opens another window.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/"runs/.mpl-cache"))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10.2, 3.55), dpi=110, facecolor='#090b18')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, facecolor='#111629')
    for i, run in enumerate(runs):
        xs, ys = series(run, metric, axis, smoothing)
        if xs:
            ax.plot(xs, ys, label=run['name'], color=COLORS[i % len(COLORS)],
                    linestyle=['-', '--', ':'][i//len(COLORS) % 3], linewidth=1.8,
                    marker='.' if len(xs) == 1 else None)
    ax.tick_params(colors='#dfe7f8', labelsize=8*text_scale)
    for spine in ax.spines.values():
        spine.set_color('#3b4563')
    ax.set_xlabel('Completed episode' if axis == 'episode' else 'Agent decisions', color='#dfe7f8', fontsize=9*text_scale)
    ax.set_ylabel(metric.replace('_',' ').title(), color='#dfe7f8', fontsize=9*text_scale)
    ax.grid(alpha=.15)
    if ax.lines:
        ax.legend(fontsize=7*text_scale, facecolor='#111629', labelcolor='white',
                  loc='upper left', ncol=2 if len(ax.lines)>4 else 1)
    else:
        ax.text(.5, .5, 'No completed data for this selection yet', transform=ax.transAxes,
                ha='center', color='white')
    fig.tight_layout(pad=1)
    output = io.BytesIO()
    canvas.print_png(output)
    return output.getvalue()
