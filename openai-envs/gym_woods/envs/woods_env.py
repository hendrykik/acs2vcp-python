import logging
import random
import sys

import gym
import numpy as np
import networkx as nx
from gym import spaces, utils

from gym_maze.internal.maze_impl import ACTION_LOOKUP, find_action_by_direction
from gym_woods.woods import Woods


class WoodsObservationSpace(gym.Space):
    """

    Mapping:
    . - path
    * - agent
    O, Q - wall
    F, G - reward
    """

    SYMBOLS = ('.', '*', 'O', 'Q', 'F', 'G')

    def seed(self, seed):
        pass

    def __init__(self, n):
        # n is the number of visible neighbour fields, typically 8
        self.n = n
        gym.Space.__init__(self, (self.n,), str)

    def sample(self):
        return tuple(random.choice(self.SYMBOLS) for _ in range(self.n))

    def contains(self, x):
        return all(elem in self.SYMBOLS for elem in x)


class AbstractWoods(gym.Env):

    def __init__(self, matrix):
        self.maze = Woods(matrix)
        self.pos_x = None
        self.pos_y = None

        self.action_space = spaces.Discrete(8)
        self.observation_space = WoodsObservationSpace(8)
        self._transitions = self._calculate_transitions()

    def reset(self):
        logging.debug('Resetting the environment')
        self._insert_animat()
        return self._observe()

    def step(self, action):
        previous_observation = self._observe()
        self._take_action(action, previous_observation)

        observation = self._observe()
        reward = self._get_reward()
        episode_over = self._is_over()

        return observation, reward, episode_over, {}

    def render(self, mode='human'):
        if mode == 'human':
            snapshot = np.copy(self.maze.matrix)
            snapshot[self.pos_y, self.pos_x] = 'X'

            sys.stdout.write("\n")
            for row in snapshot:
                sys.stdout.write(" ".join(self._render(el) for el in row))
                sys.stdout.write("\n")
            sys.stdout.flush()

        else:
            super(AbstractWoods, self).render(mode=mode)

    def _take_action(self, action, observation):
        """Executes the action inside the maze"""
        animat_moved = False
        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and not self.is_wall(observation[0]):
            self.pos_y -= 1
            animat_moved = True

            if self.pos_y < 0:
                self.pos_y = self.maze.max_y - 1

        if action_type == 'NE' and not self.is_wall(observation[1]):
            self.pos_x += 1
            self.pos_y -= 1
            animat_moved = True

            if self.pos_y < 0:
                self.pos_y = self.maze.max_y - 1

            if self.pos_x >= self.maze.max_x:
                self.pos_x = 0

        if action_type == "E" and not self.is_wall(observation[2]):
            self.pos_x += 1
            animat_moved = True

            if self.pos_x >= self.maze.max_x:
                self.pos_x = 0

        if action_type == 'SE' and not self.is_wall(observation[3]):
            self.pos_x += 1
            self.pos_y += 1
            animat_moved = True

            if self.pos_x >= self.maze.max_x:
                self.pos_x = 0

            if self.pos_y >= self.maze.max_y:
                self.pos_y = 0

        if action_type == "S" and not self.is_wall(observation[4]):
            self.pos_y += 1
            animat_moved = True

            if self.pos_y >= self.maze.max_y:
                self.pos_y = 0

        if action_type == 'SW' and not self.is_wall(observation[5]):
            self.pos_x -= 1
            self.pos_y += 1
            animat_moved = True

            if self.pos_x < 0:
                self.pos_x = self.maze.max_x - 1

            if self.pos_y >= self.maze.max_y:
                self.pos_y = 0

        if action_type == "W" and not self.is_wall(observation[6]):
            self.pos_x -= 1
            animat_moved = True

            if self.pos_x < 0:
                self.pos_x = self.maze.max_x - 1

        if action_type == 'NW' and not self.is_wall(observation[7]):
            self.pos_x -= 1
            self.pos_y -= 1
            animat_moved = True

            if self.pos_x < 0:
                self.pos_x = self.maze.max_x - 1

            if self.pos_y < 0:
                self.pos_y = self.maze.max_y - 1

        return animat_moved

    def _insert_animat(self):
        possible_coords = self.maze.possible_insertion_cords

        starting_position = random.choice(possible_coords)
        self.pos_x = starting_position[0]
        self.pos_y = starting_position[1]

    def _observe(self):
        return self.maze.perception(self.pos_x, self.pos_y)

    def _perception(self, posx, posy):
        return self.maze.perception(posx, posy)

    def _get_reward(self):
        if self.maze.is_reward(self.pos_x, self.pos_y):
            return 1000

        return 0

    def _is_over(self):
        return self.maze.is_reward(self.pos_x, self.pos_y)

    @staticmethod
    def is_wall(obs):
        return obs in ['O', 'Q']

    @staticmethod
    def _render(el):
        if el in ('O', 'Q'):
            return utils.colorize('■', 'gray')
        elif el == '.':
            return utils.colorize('□', 'white')
        elif el in ('F', 'G'):
            return utils.colorize('$', 'yellow')
        elif el == '*':
            return utils.colorize('A', 'red')
        else:
            return utils.colorize(el, 'cyan')

    def _state_action(self):
        """
        Return states and possible actions in each of them
        """
        mapping = {}

        for x, y in self.maze.possible_insertion_cords:
            [n, ne, e, se, s, sw, w, nw] = self.maze.perception(x, y)
            key = (x, y)
            mapping[key] = []

            actions_perceptions = {
                'N': n,
                'NE': ne,
                'E': e,
                'SE': se,
                'S': s,
                'SW': sw,
                'W': w,
                'NW': nw
            }

            for action, perception in actions_perceptions.items():
                if not self.is_wall(perception):
                    mapping[key].append(find_action_by_direction(action))

        # Goal state
        for key in self.maze.reward_cords:
            mapping[key] = []

        # Cast (int, int) key to (str, str)
        mapping = {(str(k[0]), str(k[1])): v for k, v in mapping.items()}

        return mapping

    def get_transitions(self):
        """
        Returns all possible transitions within the woods environment.
        Compatible with gym_maze format for knowledge calculation.
        """
        return self._transitions

    def get_goal_state(self):
        """
        Returns the perception of the goal state (first reward location).
        Compatible with gym_maze interface.
        """
        if self.maze.reward_cords:
            x, y = self.maze.reward_cords[0]
            return self._perception(x, y)
        return None

    def get_accurate_goal_state(self):
        """
        Returns the accurate goal state as perception (not coordinates).
        Compatible with gym_maze interface - returns perception of goal.
        """
        if self.maze.reward_cords:
            x, y = self.maze.reward_cords[0]
            return self._perception(x, y)
        return None

    def _calculate_transitions(self):
        """
        Calculate all possible transitions in the woods environment.
        Returns list of (position, action, next_position) tuples.
        """
        transitions = []
        g = self._create_graph()

        path_nodes = (node for node, data
                      in g.nodes(data=True) if data['type'] == 'path')

        for node in path_nodes:
            for neighbour in nx.all_neighbors(g, node):
                direction = self._distinguish_direction(node, neighbour)
                action = find_action_by_direction(direction)
                transitions.append((node, action, neighbour))

        return transitions

    def _create_graph(self):
        """
        Create a networkx graph representing the woods environment.
        """
        matrix = self.maze.matrix
        g = nx.Graph()

        # Add nodes for paths and rewards
        for y in range(self.maze.max_y):
            for x in range(self.maze.max_x):
                if matrix[y, x] == '.':
                    g.add_node((y, x), type='path')
                elif matrix[y, x] in ('F', 'G'):
                    g.add_node((y, x), type='reward')

        # Add edges between adjacent non-wall nodes
        path_nodes = [cords for cords, attribs
                      in g.nodes(data=True) if attribs['type'] == 'path']

        for node in path_nodes:
            y, x = node
            # Check all 8 directions
            neighbors = [
                (y - 1, x),     # N
                (y - 1, x + 1), # NE
                (y, x + 1),     # E
                (y + 1, x + 1), # SE
                (y + 1, x),     # S
                (y + 1, x - 1), # SW
                (y, x - 1),     # W
                (y - 1, x - 1)  # NW
            ]

            for neighbor_y, neighbor_x in neighbors:
                if 0 <= neighbor_y < self.maze.max_y and 0 <= neighbor_x < self.maze.max_x:
                    if matrix[neighbor_y, neighbor_x] in ('.', 'F', 'G'):
                        g.add_edge(node, (neighbor_y, neighbor_x))

        return g

    @staticmethod
    def _distinguish_direction(start, end):
        """
        Determine the direction from start to end position.
        Returns direction string like 'N', 'NE', 'E', etc.
        """
        direction = ''
        
        vertical_diff = end[0] - start[0]
        horizontal_diff = end[1] - start[1]

        if vertical_diff != 0:
            if vertical_diff > 0:
                direction += 'S'
            else:
                direction += 'N'

        if horizontal_diff != 0:
            if horizontal_diff > 0:
                direction += 'E'
            else:
                direction += 'W'

        return direction
