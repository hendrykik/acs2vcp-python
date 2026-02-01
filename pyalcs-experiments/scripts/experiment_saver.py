import os
import dill
from lcs.agents import Agent


class ExperimentSaver:
    def __init__(self, data_path):
        self.data_path = data_path
        pass

    def _save_data(self, data, path, file_name):
        full_dir_path = os.path.join(self.data_path, path)
        full_file_path = os.path.join(full_dir_path, f'{file_name}.dill')
        if not os.path.isdir(full_dir_path):
            os.makedirs(full_dir_path)

        dill.dump(data, open(full_file_path, 'wb'))

    def _save_agent_data(self, agent, data, path, file_name):
        path = os.path.join(type(agent).__name__, path)
        self._save_data(data, path, file_name)

    def _save_metrics(self, agent, metrics, path, metrics_name):
        self._save_agent_data(agent, metrics, path, f'metrics_{metrics_name}')

    def _save_explore_metrics(self, agent, metrics, path):
        self._save_metrics(agent, metrics, path, 'EXPLORE')

    def _save_exploit_metrics(self, agent, metrics, path):
        self._save_metrics(agent, metrics, path, 'EXPLOIT')

    def _save_population(self, agent: Agent, path):
        self._save_agent_data(agent, agent.get_population(), path,
                              'population')

    def _save_environment(self, agent, env, path):
        self._save_agent_data(agent, env, path, 'env')

    def save_experiment_data(self, agent, env, explore_metrics,
                             exploit_metrics,
                             path):
        self._save_explore_metrics(agent, explore_metrics, path)
        self._save_exploit_metrics(agent, exploit_metrics, path)
        self._save_population(agent, path)
        self._save_environment(agent, env, path)
