import os
import json
import gym

from lcs.agents import Agent
from lcs.agents.acs2vcp  import ACS2VCPv10, Configuration as CFG_ACS2VCP
from lcs.metrics import population_metrics

# Logger
import logging
import time
from scripts.experiment_saver import ExperimentSaver

logging.basicConfig(level=logging.INFO)

MAZE = "Maze4-v0"
EXPLORE_TRIALS = 500
EXPLOIT_TRIALS = 200

# The size of ER replay memory buffer
ER_BUFFER_SIZE = 10000
# The minimum number of samples of ER replay memory buffer to start replying samples (warm-up phase)
ER_BUFFER_MIN_SAMPLES = 1000
# The number of samples to be replayed druing ER phase
ER_SAMPLES_NUMBER = 8
# The number of HER new goals to be generated
HER_NEW_GOALS = 2
# The number of VCP heads
HEADS_NUMBER = 4

#######

REPEAT_START = 0
REPEAT = 30

current_dir = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_NAME = "0"
DATA_BASE_PATH = ""
DATA_PATH = os.path.join(current_dir, 'MAZE', EXPERIMENT_NAME, MAZE)

saver = ExperimentSaver(DATA_PATH)

def _save_experiment_json(agent, env, explore_metrics, exploit_metrics_1, exploit_metrics_2, exploit_metrics_3, data_path, explore_time, exploit_time):
    """Nowa funkcja do zapisywania logów w JSON."""
    log_data = {
        "agent": str(agent), 
        "environment": str(env),
        "explore_metrics": explore_metrics,
        "exploit_metrics_1": exploit_metrics_1,
        "exploit_metrics_2": exploit_metrics_2,
        "exploit_metrics_3": exploit_metrics_3,
        "explore_time": explore_time,
        "exploit_time": exploit_time 
    }
    
    full_path = DATA_PATH + '/ACS2VCP/' + data_path
        
    print("data", data_path)
    print("DATA", DATA_PATH)
        
    json_path = os.path.join(full_path, "experiment_log.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)


def _get_transitions():
    knowledge_env = gym.make(MAZE)
    transitions = knowledge_env.env.get_transitions()
    last_p = ["#", "#", "#", "#", "#", "#", "#", "#"]
    transitions = list(
        map(lambda t: [knowledge_env.env.maze.perception(t[0]) + last_p, t[1],
                       knowledge_env.env.maze.perception(t[2]) + last_p],
            transitions))

    return transitions


TRANSITIONS = _get_transitions()
TRANSITIONS_LENGTH = len(TRANSITIONS)


def _maze_knowledge(agent) -> float:
    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in agent.ensemble_heads[0].population if c.is_reliable()]

    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for p0, action, p1 in TRANSITIONS:
        if any([True for cl in reliable_classifiers
                if cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1

    return nr_correct / TRANSITIONS_LENGTH * 100.0


def _maze_metrics(agent, env):
    pop = agent.population
    metrics = {
        'knowledge': _maze_knowledge(agent)
    }
    metrics.update(population_metrics(agent.ensemble_heads[0].population, env))

    return metrics


def _run_experiment(agent: Agent, data_path=''):
    start_time = time.time()
    maze = gym.make(MAZE)
    
    agent.cfg.epsilon = 0.8
    print('przed explore ',agent.cfg.epsilon)
    explore_metrics = agent.explore(maze, EXPLORE_TRIALS)
    explore_time = time.time() - start_time
    
    agent.cfg.epsilon = 0.2
    print('przed exploit ',agent.cfg.epsilon)
        
    exploit_metrics_1 = agent.exploit(maze, EXPLOIT_TRIALS)
    
    agent.cfg.epsilon = 0.0
    print('przed 2 exploit ',agent.cfg.epsilon)

    exploit_metrics_2 = agent.exploit(maze, EXPLOIT_TRIALS)
    exploit_metrics_3 = agent.exploit(maze, EXPLOIT_TRIALS)
    
    total_time = time.time() - start_time
    exploit_time = total_time - explore_time
    
    

    saver.save_experiment_data(agent, maze, explore_metrics, 0,
                               data_path)
    
    _save_experiment_json(agent, maze, explore_metrics, exploit_metrics_1, exploit_metrics_2, exploit_metrics_3, data_path, explore_time, exploit_time)



def _run_acs2vcp_experiment(her_goals_number: int, er_samples: int, HEADS_NUMBER: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2VCP(
            classifier_length=16,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples,
            her_goals_number=her_goals_number,
            user_metrics_collector_fcn=_maze_metrics,
            epsilon=0.8,
            beta=0.05,
            gamma=0.95,
            chi=0.8,
            mu=0.3,
        )
        agent = ACS2VCPv10(cfg, ensemble_size = HEADS_NUMBER)

        _run_experiment(agent, os.path.join(f'h{her_goals_number}_e{er_samples}', f'{i}'))


_run_acs2vcp_experiment(HER_NEW_GOALS, ER_SAMPLES_NUMBER, HEADS_NUMBER)