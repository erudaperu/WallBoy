# Current experiment lab verification — 2026-11-23

The current project passes 24 tests, including resizable display controls,
saved text preferences, configurable DQN updates, model-width checkpoint
loading, reward changes, settings-form input, and graph data alignment.

A two-episode CPU smoke run with 16-neuron hidden layers verified the new
settings-to-training path and output files. This was a plumbing check, not a
learning-quality benchmark. Its exported C++ policy matched all 287 test
observations, with a maximum Q error of 4.47035e-08.

The screenshots in `assets/` show the current Pygame interface. The comparison
contains the bundled demo and two local 350-episode runs. Their recorded curves
match and therefore overlap. Only the demo run is bundled in this public repo.
The harder double-jump mode remains manual; the trained policy uses Classic.

The following sections retain the earlier game and demo verification details.

---

# Verification — November 23, 2026

## Double-jump update

The harder manual mode adds two jumps per landing, fast fall, towers, paired
barriers, wide obstacles, overhead lasers, energy orbs, score multipliers,
local best scores, a speed ramp and Nightmare difficulty. The original RL
environment and checkpoints are unchanged; no new training was performed.

Speed now increases at each 15-second level boundary rather than continuously.
Boundary checks verify the first increase, constant speed within a level,
the announcement timer, caps in both difficulties, and reset to level 1.

The suite now has **19 tests**. New checks cover double-jump limits/recharging,
fast fall, laser collisions, scoring, deterministic generation, speed caps,
real Pygame key events for both jumps/fast fall, and isolated-pattern feasibility
at the initial and maximum speeds of both difficulties. Feasibility checks
search press timings only inside tests; there is no scripted arcade bot.
These tests do not prove every possible full-course run is beatable.

Triple hurdles and rising staircases now add more hazards per group. Gaps
shrink through level 5, then have a strict floor of 660 pixels in Hard and
825 in Nightmare, calculated from maximum speed rather than current speed.
A separate search found valid routes in **12,312** combinations of pattern,
difficulty, level and approach phase, including all 170 tick offsets before
each of the first six level transitions. Accepted routes need at most 26 ticks
of preparation and 24 ticks to land after clearing; minimum group spacing
reserves 55 ticks at the speed cap (over 52 ticks after player width).
Run `PYTHONPATH=. .venv/bin/python tests/check_fairness.py --full` to repeat.
The search is a course-design check and does not train or play the game.

The challenge screen was rendered and visually inspected. Earlier measured
learning results below apply exclusively to Classic mode.

## Measured learning

- Fresh random initialization, seed 7, CPU, 350 episodes, 20,197 decisions.
- Recorded training/validation/checkpoint interval: 11.75 seconds. This
  excludes initialization, final test evaluation, and plotting.
- Best policy chosen only using validation tracks 10000–10004.
- On held-out tracks 20000–20019: random actions 1.40 mean clears;
  initial random network 1.40; selected best 22.25; latest network 23.15.
- Best completed 19/20 rounds; latest completed 20/20.
- Additional best-policy test on tracks 30000–30099: 22.11 mean clears,
  28.5405 mean seconds, 91/100 full 30-second rounds.
- Raw episode history, per-track outcomes and checkpoint metadata are in
  `runs/demo`. These finite samples do not establish perfect/general play.

## Automated checks

`python -m unittest discover -s tests -v` passed **10 tests**:

1. Collision detection and clear/collision rewards.
2. Identical seeded tracks despite different player actions.
3. Jump trajectory, landing and no midair double jump.
4. Step-based training matches tick-based playback physics and rewards.
5. Time-limit truncation is distinct from collision termination.
6. Real collected experiences change neural-network weights; saved policies
   load with exactly matching Q values.
7. Collisions remove the Bellman bootstrap; time limits preserve it.
8. Home-menu keyboard selection.
9. Space-key events reach the human player's jump physics.
10. Live-training pause, fast mode and orderly exit controls.

Also verified:

- Existing output directories are rejected without overwriting runs.
- A short live-training run saves policies, a CSV, a PNG and evaluation.
- A 180-episode live-display run completed using SDL offscreen rendering.
- Manual, race and training rendering exercised through SDL dummy/offscreen
  surfaces; generated screenshots inspected for text/layout problems.
- Playback opened through the actual X11 display and ran for 180 frames.
- Isolated environment has `include-system-site-packages = false`.
- Dependency consistency check: no broken requirements.

## C++ export

Compiled `cpp/verify.cpp` with C++17 against the generated `policy.hpp`.
All **4,606** reachable-state fixtures matched the Python action. Maximum
absolute Q-value error: **2.86102e-06** (tolerance 0.001).
Fixtures contain states reached by greedy and random policies on seeds
40000–40019. Export includes model weights, normalization descriptions,
game configuration and action cadence.

## CUDA and existing installations

The restricted process initially failed to access NVIDIA. A desktop-level
read confirmed NVIDIA GeForce RTX 4070 Laptop GPU, driver 620.70.06 and
8188 MiB VRAM. A DQN forward/backward update passed with existing PyTorch
2.10.0+cu128; the returned loss was 0.23949368298053741. No packages were
installed or changed in that environment. Full CUDA training was not run.

The new isolated CPU environment uses Python 3.13.11, PyTorch 2.15.0+cpu,
Pygame 2.6.1, NumPy 2.5.5 and Matplotlib 3.11.2. Full versions are recorded
in `requirements-tested.txt`. No system packages, drivers, existing AI
environments or original C++ Wallboy files were changed.

## Scope still unverified

- A physical keyboard play session by a human; key handling was automated.
- Full integration with the existing C++ Wallboy program. Only the standalone
  C++ policy inference is verified.
- CUDA end-to-end convergence/performance, cross-machine bitwise reproduction,
  arbitrary changed game physics, and unlimited-duration survival.

Screenshots are actual Pygame renders, not design mockups. `desktop-playback.png`
comes from the X11 playback check; other screenshots use offscreen rendering.
