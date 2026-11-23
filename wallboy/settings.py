"""One list drives the CLI, training form, defaults, and learning explanations."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    default: object
    minimum: float
    maximum: float
    help: str
    group: str


OPTIONS = [
    Option('episodes', 'Episodes', 350, 1, 100000, 'One episode is one attempt until a crash or time limit. More attempts give the agent more experience, but take longer. Try 50, then 350.', 'Run'),
    Option('seed', 'Random seed', 7, 0, 9999, 'Sets the starting randomness for weights, exploration and tracks. Repeat a seed to compare settings; also try several seeds because one lucky run is not proof.', 'Run'),
    Option('max_ticks', 'Episode ticks', 1800, 60, 36000, '60 ticks = one second. 1800 means a 30-second attempt. Longer episodes change the possible reward, so compare survival and obstacles too.', 'Run'),
    Option('eval_every', 'Evaluate every', 20, 1, 10000, 'Every this many episodes, test without random actions on five fixed validation tracks. This selects best.pt. Validation shows what the network learned, without exploration noise.', 'Run'),
    Option('learning_rate', 'Learning rate', .0005, .000001, .1, 'How big each weight adjustment is. Too large can make learning unstable; too small can be slow. Try 0.0001 versus 0.0005 with the same seed.', 'Network'),
    Option('gamma', 'Future value', .99, 0, 1, 'Discount factor: how much future rewards matter. 0 only values the next reward; 0.99 values longer-term survival. The DQN target is reward + gamma * future Q.', 'Network'),
    Option('hidden_size', 'Neurons per layer', 64, 8, 256, 'The two hidden layers each have this many neurons. More neurons can represent more patterns, but cost more computation. Inputs stay 7 and actions stay wait/jump.', 'Network'),
    Option('gradient_clip', 'Gradient limit', 10., .1, 100, 'Limits the overall size of gradients before a weight update. This helps prevent a surprising batch from producing a huge update. It is not a reward cap.', 'Network'),
    Option('batch_size', 'Batch size', 128, 1, 2048, 'How many remembered decisions are sampled for one update. Larger batches smooth the update but use more computation. Must fit inside replay memory.', 'Memory'),
    Option('replay_size', 'Memory capacity', 50000, 1, 1000000, 'Maximum stored experiences: observation, action, reward, next observation, and collision flag. When full, the oldest experience is replaced.', 'Memory'),
    Option('warmup_steps', 'Warm-up decisions', 1000, 1, 1000000, 'Collect this many experiences before learning. This avoids learning from only a handful of early moves. Learning also waits until a whole batch is available.', 'Memory'),
    Option('target_update', 'Target update', 500, 1, 100000, 'Copy online weights to the target network every this many learning updates. A slower-changing target makes the moving prediction goal more stable.', 'Memory'),
    Option('epsilon_start', 'Starting exploration', 1., 0, 1, 'Probability of a random action at the start. 1 = completely random; 0 = always trust current Q values. Random actions help discover useful jumps.', 'Exploration'),
    Option('epsilon_end', 'Final exploration', .03, 0, 1, 'Smallest random-action probability after the exploration schedule. 0.03 keeps trying random actions 3% of the time. Validation always uses zero exploration.', 'Exploration'),
    Option('explore_steps', 'Exploration decay', 20000, 1, 10000000, 'Number of decisions used to decrease exploration linearly from start to end. Too short may stop discovery early. Too long makes training rewards noisy for longer.', 'Exploration'),
    Option('alive_reward', 'Alive reward / tick', .02, -1, 1, 'Reward for each surviving physics tick. A decision spans four ticks. Increasing this tells the agent that staying alive matters more.', 'Rewards'),
    Option('clear_reward', 'Obstacle reward', 3., -20, 20, 'Reward when an obstacle passes behind the player. A larger bonus emphasizes making progress. This changes the meaning of total reward between runs.', 'Rewards'),
    Option('jump_cost', 'Jump cost', .03, 0, 5, 'Penalty for starting a jump, not for ineffective presses while airborne. Encourages avoiding unnecessary jumps. Too large can discourage jumping at all.', 'Rewards'),
    Option('death_cost', 'Crash cost', 5., 0, 100, 'Penalty when a collision ends the episode. Larger values emphasize avoiding crashes. Rewards are your teaching signal, not scripted jump rules.', 'Rewards'),
]
GROUPS = ['Run', 'Network', 'Memory', 'Exploration', 'Rewards']
DEFAULTS = {o.key: o.default for o in OPTIONS}


def validate(values):
    result = {}
    for option in OPTIONS:
        try:
            value = float(values.get(option.key, option.default))
        except (TypeError, ValueError):
            raise ValueError(f'{option.label}: enter a number') from None
        if not math.isfinite(value) or not option.minimum <= value <= option.maximum:
            raise ValueError(f'{option.label}: use {option.minimum:g} to {option.maximum:g}')
        if type(option.default) is int:
            if not value.is_integer():
                raise ValueError(f'{option.label}: use a whole number')
            value = int(value)
        result[option.key] = value
    if result['batch_size'] > result['replay_size'] or result['warmup_steps'] > result['replay_size']:
        raise ValueError('Memory capacity must cover both batch size and warm-up decisions')
    if result['epsilon_end'] > result['epsilon_start']:
        raise ValueError('Final exploration must be no greater than starting exploration')
    return result


def exploration(step, values):
    fraction = min(1., step / values['explore_steps'])
    return values['epsilon_start'] + fraction*(values['epsilon_end']-values['epsilon_start'])
