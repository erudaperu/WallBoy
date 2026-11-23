"""Optional design check. Run: PYTHONPATH=. .venv/bin/python tests/check_fairness.py --full

Search for a valid two-press path, with at most 26 ticks of preparation and
24 ticks to land after clearing. The generator leaves at least 55 ticks of
travel between groups; subtracting player width still leaves over 52 ticks.
This search is a TEST, never a player controller or an RL training source.
"""
import argparse
import copy
from wallboy.challenge import Challenge, PATTERNS, make_pattern

def route(kind, nightmare, elapsed):
    baseline=Challenge(42,nightmare)
    baseline.obstacles=make_pattern(kind,680)
    baseline.orbs=[]
    baseline.next_pattern=100000
    baseline.ticks=elapsed
    entry=None
    # Measure first overlap tick using the exact level-dependent scroll speed.
    probe=copy.deepcopy(baseline)
    probe.y=10000
    for t in range(150):
        probe.tick()
        if probe.obstacles[0].x < probe.config.player_x+probe.config.player_width:
            entry=t
            break
    for first in range(max(0,entry-26),entry+1):
        for delay in range(1,38):
            env=copy.deepcopy(baseline)
            passed=None
            for t in range(170):
                env.tick(int(kind!='laser' and t in (first,first+delay)))
                if env.dead: break
                if env.cleared==len(PATTERNS[kind]):
                    if passed is None: passed=t
                    if env.y==0:
                        if t-passed<=24: return (entry-first,t-passed,first,delay)
                        break
    return None
if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    worst=[0,0]
    count = 0
    # Outside these boundary phases, speed stays constant for the whole check.
    times = ([level*900 + phase for level in range(6) for phase in [0, *range(730,900)]]
             if args.full else [level*900 + phase for level in range(6)
                                for phase in (0, 830, 860, 880, 899)])
    for mode in (False,True):
        for elapsed in times:
            for kind in PATTERNS:
                found=route(kind,mode,elapsed)
                if found is None: raise AssertionError((kind,mode,elapsed))
                worst=[max(worst[0],found[0]),max(worst[1],found[1])]
                count += 1
    print('PASS:',count,'cases. Maximum preparation/recovery ticks:',worst)
