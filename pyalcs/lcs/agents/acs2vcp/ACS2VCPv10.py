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
# from lcs.agents.acs2her.ReplayBuffer import ReplayBuffer
from lcs.strategies.action_selection.BestAction import BestAction
from time import sleep
import time as time_lib

logger = logging.getLogger(__name__)

class ACS2VCPv10(Agent):

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

    def _run_trial_explore(self, env, time,
                           current_trial=None) -> TrialMetrics:

        logger.debug("** Running trial explore ** ")
        trial_steps_all = {}
        
        for i in range(len(self.ensemble_heads)):
            trial_steps = []
            state = env.reset()
            last_reward = 0
            done = False

            self.ensemble_heads[i].main_goal = Perception(env.get_accurate_goal_state())

            while not done:
                state = Perception(state)
                assert len(state) * 2 == self.ensemble_heads[i].cfg.classifier_length

                match_set = self.ensemble_heads[i].population.form_match_set(state)
                action = self.ensemble_heads[i].cfg.action_selector(match_set)
                logger.debug("\tExecuting action: [%d]", action)

                prev_state = Perception(state)
                raw_state, last_reward, done, _ = env.step(action)
                state = Perception(raw_state)
                trial_steps.append(
                    [prev_state, action, last_reward, state, done])
            
            for index, step in enumerate(trial_steps):
                state, action, reward, next_state, done = step
                new_exp = []

                new_exp.append(ReplayMemorySample(
                    ACS2HER.state_goal_concat(state, self.ensemble_heads[i].main_goal),
                    action,
                    reward,
                    ACS2HER.state_goal_concat(next_state, self.ensemble_heads[i].main_goal),
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
                trial_steps_all[i] = trial_steps
         
        # for _, (head, trail_steps) in enumerate(trial_steps_all.items()):
        #     for i in range(len(trail_steps)):
        #         self.learn(time, i, head)
        
        self.learn_new_buffer(time, trial_steps_all)
        
            
        # for i in range(len(self.ensemble_heads)):
        #    print(f"Head {i} → Population: {len(self.ensemble_heads[i].population)}")

        
        return TrialMetrics(len(trial_steps), last_reward)

    def _run_trial_exploit(self, env, time=None,
                           current_trial=None) -> TrialMetrics:

        logger.debug("** Running trial exploit **")
                            
        steps = 0
        state = Perception(env.reset())
        
        self.ensemble_heads[0].main_goal = Perception(env.get_accurate_goal_state())        

        last_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:
            self.ensemble_heads[0].main_goal = Perception(env.get_accurate_goal_state())
            
            match_set = self.ensemble_heads[0].population.form_match_set(
                ACS2HER.state_goal_concat(state, self.ensemble_heads[0].main_goal))
            
            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    self.ensemble_heads[0].cfg.beta,
                    self.ensemble_heads[0].cfg.gamma)

            action = BestAction(
                all_actions=self.ensemble_heads[0].cfg.number_of_possible_actions)(match_set)
            action_set = match_set.form_action_set(action)

            state, last_reward, done, _ = env.step(action)
            state = Perception(state)

            if done:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.ensemble_heads[0].cfg.beta, self.ensemble_heads[0].cfg.gamma)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def learn_new_buffer(self, time_arg, trial_steps_all):
        max_steps = max(len(steps) for steps in trial_steps_all.values()) if trial_steps_all else 0
        
        for trail_num in range(max_steps):
            all_learning_tasks = []
            all_exp_ind = {}
            
            for agent_id, trail_steps in trial_steps_all.items(): 
                if trail_num < len(trail_steps):
                    experiences, indices, weights = self.memory.sample()
                    all_learning_tasks.append((
                        time_arg,
                        trail_num,
                        experiences,
                        agent_id
                    ))
                    for exp, idx in zip(experiences, indices):
                        exp_key = (tuple(exp.state), exp.action, exp.reward, 
                                tuple(exp.next_state), exp.done)
                        if exp_key not in all_exp_ind:
                            all_exp_ind[exp_key] = (exp, idx)
                    
            if len(all_learning_tasks) > 0:
                start_time = time_lib.time()
                
                for time, steps, experiences, agent_id in all_learning_tasks:
                    self.learn(agent_id, time, steps, experiences)
                
                # print('Uczenie:  ', time_lib.time() - start_time)
                    
                start_time = time_lib.time()
                                                                    
                new_sigmas = []
                new_prior = []
                final_indices = []
                
                for exp, idx in all_exp_ind.values():
                    q_vals = [self.compute_q_prediction(head, exp.state, exp.action)
                                for head in self.ensemble_heads]
                    sigma2 = np.var(q_vals)
                    new_sigmas.append(sigma2)
                    self.memory.sigma2_max = max(self.memory.sigma2_max, sigma2)
                    p = (self.memory.sigma2_max - sigma2 + self.memory.eps) ** self.memory.T
                    new_prior.append(p)
                    final_indices.append(idx)
                
                self.memory.update_priorities(final_indices, new_sigmas, new_prior)
                
                # print('Zmiana wybranych priorytetów:  ', time_lib.time() - start_time)
                
            

    def learn(self, i, time, steps, experiences):
        for idx, exp in enumerate(experiences):
            er_match_set = self.ensemble_heads[i].population.form_match_set(
                exp.state)
            er_action_set = er_match_set.form_action_set(
                exp.action)
            er_next_match_set = self.ensemble_heads[i].population.form_match_set(
                exp.next_state)
            ClassifiersList.apply_alp(
                self.ensemble_heads[i].population,
                er_next_match_set,
                er_action_set,
                exp.state,
                exp.action,
                exp.next_state,
                time + steps,
                self.ensemble_heads[i].cfg.theta_exp,
                self.ensemble_heads[i].cfg)
            ClassifiersList.apply_reinforcement_learning(
                er_action_set,
                exp.reward,
                0 if exp.done
                else er_next_match_set.get_maximum_fitness(),
                self.ensemble_heads[i].cfg.beta,
                self.ensemble_heads[i].cfg.gamma
            )
            if self.ensemble_heads[i].cfg.do_ga:
                ClassifiersList.apply_ga(
                    time + steps,
                    self.ensemble_heads[i].population,
                    ClassifiersList() if exp.done else er_next_match_set,
                    er_action_set,
                    exp.next_state,
                    self.ensemble_heads[i].cfg.theta_ga,
                    self.ensemble_heads[i].cfg.mu,
                    self.ensemble_heads[i].cfg.chi,
                    self.ensemble_heads[i].cfg.theta_as,
                    self.ensemble_heads[i].cfg.do_subsumption,
                    self.ensemble_heads[i].cfg.theta_exp
                    )

    def sample_goals(self, trial_steps, index):
        steps = []
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
