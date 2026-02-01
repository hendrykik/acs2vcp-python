import logging
import random
import numpy as np
from lcs import Perception
from lcs.agents.Agent import Agent
from lcs.agents.Agent import TrialMetrics
from lcs.agents.acs2 import ClassifiersList
from lcs.agents.acs2er import ReplayMemorySample
from lcs.agents.acs2vcp.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from lcs.agents.acs2her import Configuration
from lcs.agents.acs2her import ACS2HER
from lcs.strategies.action_selection.BestAction import BestAction
from multiprocessing import Pool
import copy
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

def learning_worker_with_indices(args):
    """
    Enhanced learning worker that returns experiences and indices for priority updates
    Args: (agent_config, population_copy, experiences, time, steps, agent_id, indices)
    Returns: (agent_id, updated_population, experiences, indices)
    """
    try:
        agent_config, population_copy, time, agent_id, steps, memory = args        
        temp_agent = ACS2HER(agent_config, population_copy)   
        experiences_n = []
        indices_n = []
        
        # for steps in range(trail_steps):
        
        experiences, indices, weights = memory.sample()
    
        
        for idx, exp in enumerate(experiences):
            er_match_set = temp_agent.population.form_match_set(exp.state)
            er_action_set = er_match_set.form_action_set(exp.action)
            er_next_match_set = temp_agent.population.form_match_set(exp.next_state)
            
            ClassifiersList.apply_alp(
                temp_agent.population, er_next_match_set, er_action_set,
                exp.state, exp.action, exp.next_state, time + steps,
                temp_agent.cfg.theta_exp, temp_agent.cfg)
                
            ClassifiersList.apply_reinforcement_learning(
                er_action_set, exp.reward,
                0 if exp.done else er_next_match_set.get_maximum_fitness(),
                temp_agent.cfg.beta, temp_agent.cfg.gamma)
                
            if temp_agent.cfg.do_ga:
                ClassifiersList.apply_ga(
                    time + steps, temp_agent.population,
                    ClassifiersList() if exp.done else er_next_match_set, er_action_set,
                    exp.next_state, temp_agent.cfg.theta_ga,
                    temp_agent.cfg.mu, temp_agent.cfg.chi,
                    temp_agent.cfg.theta_as, temp_agent.cfg.do_subsumption,
                    temp_agent.cfg.theta_exp)
            experiences_n.extend(experiences)
            indices_n.extend(indices)
            
        return agent_id, temp_agent.population, experiences_n, indices_n
                    
    except Exception as e:
        logger.error(f"Learning worker {agent_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return agent_id, None, [], []

def explore_environment_worker(args):
    """
    Worker function for environment exploration
    Args: (env_state, agent_config, population_copy, agent_id, steps_limit)
    Returns: (agent_id, trial_steps, goal_state)
    """
    try:
        env_state, agent_config, population_copy, agent_id, steps_limit = args
        
        # Recreate environment and agent in process
        agent_env = copy.deepcopy(env_state)
        temp_agent = ACS2HER(agent_config, population_copy)
        
        trial_steps = []
        state = agent_env.reset()
        last_reward = 0
        done = False
        steps = 0

        # Set the main goal (exactly like in sequential version)
        temp_agent.main_goal = Perception(agent_env.get_accurate_goal_state())

        while not done and steps < steps_limit:
            state = Perception(state)
            assert len(state) * 2 == temp_agent.cfg.classifier_length

            match_set = temp_agent.population.form_match_set(state)
            action = temp_agent.cfg.action_selector(match_set)
            
            prev_state = Perception(state)
            raw_state, last_reward, done, _ = agent_env.step(action)
            state = Perception(raw_state)
            
            trial_steps.append([prev_state, action, last_reward, state, done])
            steps += 1
        
        return agent_id, trial_steps, temp_agent.main_goal, last_reward
        
    except Exception as e:
        logger.error(f"Worker {agent_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return agent_id, [], None, 0

class ACS2VCPv6(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None,
                 ensemble_size: int = 4,
                 buffer_eps: float = 1e-6,
                 buffer_T: float = 1.0) -> None:
        base_population = population or ClassifiersList()
        
        self.memory_raw = []
        self.memory = PrioritizedReplayBuffer(
            max_size=cfg.er_buffer_size,
            batch_size=cfg.er_samples_number,
            eps=buffer_eps,
            T=buffer_T
        )
        self.ensemble_heads = [
            ACS2HER(cfg, base_population)
            for _ in range(ensemble_size)
        ]
        self.cfg = cfg
        self.population = base_population

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg
    
    def compute_q_prediction(self, agent, state, action):
        match_set = agent.population.form_match_set(state)
        action_set = match_set.form_action_set(action)
        return max((cl.fitness for cl in action_set), default=0.0)

    def add_with_vcp(self, mini_batch_sample):
        for sample in mini_batch_sample:
            q_vals = [self.compute_q_prediction(head, sample.state, sample.action)
                for head in self.ensemble_heads]
            sigma = np.var(q_vals)
            self.memory.sigma2_max = max(self.memory.sigma2_max, sigma)
            
            p = (self.memory.sigma2_max - sigma + self.memory.eps) ** self.memory.T
            self.memory.add(sample, sigma, p)

    def _run_trial_explore(self, env, time, current_trial=None) -> TrialMetrics:
        """Multiprocessing version that maintains exact same logic as sequential"""
        logger.debug("** Running trial explore (multiprocessing) ** ")
        
        # Prepare arguments for workers - include all necessary data
        args = [
            (env, self.cfg, self.ensemble_heads[i].population, i, 1000) 
            for i in range(len(self.ensemble_heads))
        ]
        
        trial_steps_all = {}
        last_rewards = {}
        
        try:
            # Use multiprocessing pool
            with Pool(processes=min(len(self.ensemble_heads), 4)) as pool:
                results = pool.map(explore_environment_worker, args)
                
            # Process results and update main ensemble heads
            for agent_id, trial_steps, goal_state, last_reward in results:
                if trial_steps:
                    # Set the goal in main ensemble head to match worker
                    self.ensemble_heads[agent_id].main_goal = goal_state
                    trial_steps_all[agent_id] = trial_steps
                    last_rewards[agent_id] = last_reward
                    
        except Exception as e:
            logger.error(f"Multiprocessing failed, falling back to sequential: {e}")
            return self._run_trial_explore_sequential(env, time, current_trial)
        
        # Process experiences for each agent (exactly like sequential version)
        for agent_id, trial_steps in trial_steps_all.items():
            for index, step in enumerate(trial_steps):
                state, action, reward, next_state, done = step
                new_exp = []

                new_exp.append(ReplayMemorySample(
                    ACS2HER.state_goal_concat(state, self.ensemble_heads[agent_id].main_goal),
                    action,
                    last_rewards[agent_id],  # Use last_reward like in sequential version
                    ACS2HER.state_goal_concat(next_state, self.ensemble_heads[agent_id].main_goal),
                    done))

                additional_goals = self.sample_goals(trial_steps, index)

                for goal in additional_goals:
                    new_reward = self.reward_function(next_state, goal)

                    new_exp.append(ReplayMemorySample(
                        ACS2HER.state_goal_concat(state, goal),
                        action,
                        new_reward,
                        ACS2HER.state_goal_concat(next_state, goal),
                        False))

                self.add_with_vcp(new_exp)
        
        self.learn_parallel(time, trial_steps_all)
        
        print(current_trial)
        for i in range(len(self.ensemble_heads)):
            print(f"Head {i} → Population: {len(self.ensemble_heads[i].population)}")
        
        if trial_steps_all and last_rewards:
            last_agent_id = max(trial_steps_all.keys())
            last_trial_steps = trial_steps_all[last_agent_id]
            last_reward = last_rewards[last_agent_id]
            return TrialMetrics(len(last_trial_steps), last_reward)
        else:
            return TrialMetrics(0, 0)

    def learn_parallel(self, time, trial_steps_all):
        """Parallel learning"""
        try:
            all_experiences = [] 
            all_indices = []
            
            for trail_num in range(50):
                all_learning_tasks = []
                for _, (head, trail_steps) in enumerate(trial_steps_all.items()):
                    if trail_num <= len(trail_steps):
                        all_learning_tasks.append((
                            self.cfg,
                            self.ensemble_heads[head].population,
                            time,
                            head,
                            trail_num,
                            self.memory
                        ))
                        
                if len(all_learning_tasks) > 0:
                    with Pool(processes=min(len(all_learning_tasks), 4)) as pool:
                        results = pool.map(learning_worker_with_indices, all_learning_tasks)
                    
                    for agent_id, updated_population, experiences_n, indices_n in results:
                        self.ensemble_heads[agent_id] = ACS2HER(self.cfg, updated_population)
                        all_experiences.extend(experiences_n)
                        all_indices.extend(indices_n)
                                                        
                    unique_exp_idx = {}
                
                    for i, exp in enumerate(all_experiences):
                        exp_key = (tuple(exp.state), exp.action, exp.reward, tuple(exp.next_state), exp.done)
                        if exp_key not in unique_exp_idx:
                            unique_exp_idx[exp_key] = (exp, all_indices[i])
                    
                    new_sigmas = []
                    new_prior = []
                    final_indices = []
                    
                    for exp, idx in unique_exp_idx.values():
                        q_vals = [self.compute_q_prediction(head, exp.state, exp.action)
                                    for head in self.ensemble_heads]
                        sigma2 = np.var(q_vals)
                        new_sigmas.append(sigma2)
                        self.memory.sigma2_max = max(self.memory.sigma2_max, sigma2)
                        p = (self.memory.sigma2_max - sigma2 + self.memory.eps) ** self.memory.T
                        new_prior.append(p)
                        final_indices.append(idx)
                    
                    self.memory.update_priorities(final_indices, new_sigmas, new_prior)
                    
        except Exception as e:
            logger.error(f"Parallel learning failed, falling back to sequential: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to sequential learning
            for agent_id, trial_steps in trial_steps_all.items():
                for i in range(len(trial_steps)):
                    self.learn(time, i, agent_id)

    def _run_trial_exploit(self, env, time=None, current_trial=None) -> TrialMetrics:
        logger.debug("** Running trial exploit **")
                
        for i in range(len(self.ensemble_heads)):  # Fixed: was range(self.ensemble_heads)
            steps = 0
            state = Perception(env.reset())
            last_reward = 0
            action_set = ClassifiersList()
            done = False

            while not done:
                match_set = self.ensemble_heads[i].population.form_match_set(
                    ACS2HER.state_goal_concat(state, self.ensemble_heads[i].main_goal))

                if steps > 0:
                    ClassifiersList.apply_reinforcement_learning(
                        action_set, last_reward, match_set.get_maximum_fitness(),
                        self.ensemble_heads[i].cfg.beta, self.ensemble_heads[i].cfg.gamma)

                action = BestAction(
                    all_actions=self.ensemble_heads[i].cfg.number_of_possible_actions)(match_set)
                action_set = match_set.form_action_set(action)

                state, last_reward, done, _ = env.step(action)
                state = Perception(state)

                if done:
                    ClassifiersList.apply_reinforcement_learning(
                        action_set, last_reward, 0, 
                        self.ensemble_heads[i].cfg.beta, self.ensemble_heads[i].cfg.gamma)

                steps += 1

        return TrialMetrics(steps, last_reward)

    def learn(self, time, steps, i):
        experiences, indices, weights = self.memory.sample()
        for idx, exp in enumerate(experiences):
            er_match_set = self.ensemble_heads[i].population.form_match_set(exp.state)
            er_action_set = er_match_set.form_action_set(exp.action)
            er_next_match_set = self.ensemble_heads[i].population.form_match_set(exp.next_state)
            
            ClassifiersList.apply_alp(
                self.ensemble_heads[i].population, er_next_match_set, er_action_set,
                exp.state, exp.action, exp.next_state, time + steps,
                self.ensemble_heads[i].cfg.theta_exp, self.ensemble_heads[i].cfg)
                
            ClassifiersList.apply_reinforcement_learning(
                er_action_set, exp.reward,
                0 if exp.done else er_next_match_set.get_maximum_fitness(),
                self.ensemble_heads[i].cfg.beta, self.ensemble_heads[i].cfg.gamma)
                
            if self.ensemble_heads[i].cfg.do_ga:
                ClassifiersList.apply_ga(
                    time + steps, self.ensemble_heads[i].population,
                    ClassifiersList() if exp.done else er_next_match_set, er_action_set,
                    exp.next_state, self.ensemble_heads[i].cfg.theta_ga,
                    self.ensemble_heads[i].cfg.mu, self.ensemble_heads[i].cfg.chi,
                    self.ensemble_heads[i].cfg.theta_as, self.ensemble_heads[i].cfg.do_subsumption,
                    self.ensemble_heads[i].cfg.theta_exp)
                
        new_sigmas = []
        new_prior = []
        for exp in experiences:
            q_vals = [self.compute_q_prediction(head, exp.state, exp.action)
                    for head in self.ensemble_heads]
            sigma2 = np.var(q_vals)
            new_sigmas.append(sigma2)
            self.memory.sigma2_max = max(self.memory.sigma2_max, sigma2)
            p = (self.memory.sigma2_max - sigma2 + self.memory.eps) ** self.memory.T
            new_prior.append(p)
        self.memory.update_priorities(indices, new_sigmas, new_prior)

    def sample_goals(self, trial_steps, index):
        steps_taken = len(trial_steps)
        k = min(self.ensemble_heads[0].cfg.her_goals_number, steps_taken - index)
        steps = random.sample(trial_steps[index:], k=k) if k > 0 else []
        return [s[3] for s in steps]

    def reward_function(self, state, new_goal):
        if self.cfg.her_reward_generator is None:
            return 1 if state == new_goal else 0
        else:
            return self.cfg.her_reward_generator(state, new_goal)

    @staticmethod
    def state_goal_concat(state: Perception, goal: Perception) -> Perception:
        return Perception(tuple(state) + tuple(goal))