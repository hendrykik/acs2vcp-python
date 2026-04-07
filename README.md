# Anticipatory Learning Classifier 2 with Value Consistency Prioritization implemented in Python

Repository contains implementation of agents from the family of Learning Classifier Systems such as ACS or ACS2, enhanced with techniques like Experience Replay, Hindsight Experience Replay and a new technique - Value Consistency Prioritization.

Value Consistency Prioritization is based on the paper by Claudia Russo, Daniela Barni, Ioana Zagrean and Francesca Danioni:
https://www.mdpi.com/1210682

## Repository Structure

This repository contains:
- **openai-envs** - Gymnasium library with added new environments ([Gymnasium](https://gymnasium.farama.org))
- **pyalcs** - LCS agents with new ACS2VCP agent ([Documentation](https://pyalcs.readthedocs.io/en/latest/), [GitHub](https://github.com/ParrotPrediction/pyalcs))
- **pyalcs-experiments** - Scripts and notebooks with experiments

## Available Agents

| Agent   | Description                                      |
|---------|--------------------------------------------------|
| ACS2    | Anticipatory Learning Classifier System 2        |
| ACS2ER  | ACS2 with Experience Replay                      |
| ACS2HER | ACS2 with Hindsight Experience Replay            |
| ACS2VCP | ACS2 with Value Consistency Prioritization       |

Each agent has experiment scripts for Maze4, Maze5 and Maze7 environments in `pyalcs-experiments/scripts/`.

## Installation

Create conda environment and install local packages:
```bash
cd pyalcs-experiments
conda env create --file environment-base.yml
conda activate pyalcs-experiments
conda env update --file environment-base.yml --prune
cd ..

pip install -e ./pyalcs
pip install -e ./openai-envs
```

## Quick Start

Minimal example using ACS2 on a maze environment:
```python
import gym
import gym_maze
from lcs.agents.acs2 import ACS2, Configuration

cfg = Configuration(
    classifier_length=8,
    number_of_possible_actions=8,
    epsilon=0.8,
    beta=0.05,
    gamma=0.95,
)
agent = ACS2(cfg)
maze = gym.make('Maze4-v0')

# Explore: agent learns the environment model
explore_metrics = agent.explore(maze, 500)

# Exploit: agent uses learned knowledge (no exploration)
exploit_metrics = agent.exploit(maze, 200)

# Access learned classifiers
population = agent.get_population()
```

## Configuration

### Base parameters (all agents)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `classifier_length` | required | Length of condition/effect strings |
| `number_of_possible_actions` | required | Number of possible actions |
| `epsilon` | 0.5 | Exploration probability (random action rate) |
| `beta` | 0.05 | Learning rate |
| `gamma` | 0.95 | Discount factor |
| `chi` | 0.8 | Crossover probability |
| `mu` | 0.3 | Mutation probability |
| `do_ga` | False | Enable genetic generalization |
| `do_pee` | False | Enable Probability-Enhanced Effects |
| `metrics_trial_frequency` | 1 | Collect metrics every N trials |
| `user_metrics_collector_fcn` | None | Custom metrics callback `fn(agent, env) -> dict` |

### ACS2ER additional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `er_buffer_size` | 10000 | Replay buffer size |
| `er_min_samples` | 1000 | Min samples before replay starts |
| `er_samples_number` | 3 | Samples replayed per step |

### ACS2HER additional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `er_buffer_size` | 10000 | Replay buffer size |
| `er_samples_number` | 8 | Samples replayed per step |
| `her_goals_number` | 3 | HER goals to generate per transition |
| `her_strategy` | None | Goal selection: `'final'` or `'future'` |

### ACS2VCP

Uses the same configuration as ACS2HER. The ensemble size is passed to the constructor:
```python
agent = ACS2VCPv10(cfg, ensemble_size=4)
```

## Using Other Agents

All agents follow the same pattern. Key differences are imports, configuration parameters, and `classifier_length` (16 for HER/VCP because the goal state is appended to perception).

**ACS2ER:**
```python
from lcs.agents.acs2er import ACS2ER, Configuration

cfg = Configuration(
    classifier_length=8,
    number_of_possible_actions=8,
    er_buffer_size=10000,
    er_min_samples=1000,
    er_samples_number=8,
    epsilon=0.8, beta=0.05, gamma=0.95,
)
agent = ACS2ER(cfg)
```

**ACS2HER:**
```python
from lcs.agents.acs2her import ACS2HER, Configuration

cfg = Configuration(
    classifier_length=16,
    number_of_possible_actions=8,
    er_buffer_size=10000,
    er_samples_number=8,
    her_goals_number=2,
    epsilon=0.8, beta=0.05, gamma=0.95,
)
agent = ACS2HER(cfg)
```

**ACS2VCP:**
```python
from lcs.agents.acs2vcp import ACS2VCPv10, Configuration

cfg = Configuration(
    classifier_length=16,
    number_of_possible_actions=8,
    er_buffer_size=10000,
    er_min_samples=1000,
    er_samples_number=8,
    her_goals_number=2,
    epsilon=0.8, beta=0.05, gamma=0.95,
)
agent = ACS2VCPv10(cfg, ensemble_size=4)
```

## Running Experiments

Experiment scripts are in `pyalcs-experiments/scripts/` with naming convention `run_{agent}_{maze}.py`.

```bash
cd pyalcs-experiments
export PYTHONPATH=$(pwd)

# Run ACS2 on Maze4
python scripts/ACS2/run_acs2_maze4.py

# Run ACS2VCP on Maze7
python scripts/ACS2VCP/run_acs2vcp_maze7.py
```

Each experiment runs 30 repeats with three phases:
1. **Explore** (500 trials, epsilon=0.8) - agent learns the environment
2. **Exploit** (200 trials, epsilon=0.2) - evaluation with mild exploration
3. **Exploit** (2x200 trials, epsilon=0.0) - pure greedy evaluation

Results are saved as `experiment_log.json` and `.dill` files under `scripts/{AGENT}/MAZE/`.

## Environments

The maze environments are OpenAI Gym-compatible:
```python
import gym
import gym_maze

maze = gym.make('Maze4-v0')  # also: Maze5-v0, Maze7-v0
```

| Property | Value |
|----------|-------|
| Actions | 8 discrete (N, NE, E, SE, S, SW, W, NW) |
| Observation | 8-element perception of surrounding cells |
| Reward | +1000 at goal, 0 otherwise |
| Max steps | 50 per episode |

Additional environments (grids, corridors, woods, etc.) are registered in `openai-envs/` - see `gym_maze/__init__.py` for the full list.

## Metrics

`agent.explore()` and `agent.exploit()` return a list of dicts per trial:

| Key | Description |
|-----|-------------|
| `trial` | Trial number |
| `steps_in_trial` | Steps taken |
| `reward` | Reward received |
| `perf_time` | Wall clock time (seconds) |

Custom metrics can be added via `user_metrics_collector_fcn` in the configuration. The experiment scripts add `knowledge` (% of correctly anticipated transitions) and population statistics.

## Citation

If you use this library, please cite this paper:

```text
Jan Zemło and Olgierd Unold. 2026. Value Consistency Prioritization for Accelerating 
Knowledge Discovery in Sparse Reward Anticipatory Classifier Systems. In Proceedings 
of the Genetic and Evolutionary Computation Conference Companion. ACM. 
```
