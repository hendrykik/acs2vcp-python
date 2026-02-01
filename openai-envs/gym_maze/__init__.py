from gym.envs.registration import register

from gym_maze.common import MAZE_PATH as PATH_MAPPING  # noqa: F401
from gym_maze.common import MAZE_REWARD as REWARD_MAPPING  # noqa: F401
from gym_maze.common import MAZE_WALL as WALL_MAPPING  # noqa: F401
from gym_maze.maze import Maze  # noqa: F401
from gym_maze.rotating_maze import RotatingMaze  # noqa: F401

register(
    id='Cassandra4x4-v0',
    entry_point='gym_maze.envs:Cassandra4x4',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Littman57-v0',
    entry_point='gym_maze.envs:Littman57',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Littman89-v0',
    entry_point='gym_maze.envs:Littman89',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeMA-v0',
    entry_point='gym_maze.envs:MazeMA',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MiyazakiA-v0',
    entry_point='gym_maze.envs:MiyazakiA',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MiyazakiB-v0',
    entry_point='gym_maze.envs:MiyazakiB',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Woods1-v0',
    entry_point='gym_woods.envs:Woods1',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Woods14-v0',
    entry_point='gym_woods.envs:Woods14',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Woods100-v0',
    entry_point='gym_woods.envs:Woods100',
    max_episode_steps=500,
    nondeterministic=True
)

register(
    id='Woods101-v0',
    entry_point='gym_woods.envs:Woods101',
    max_episode_steps=500,
    nondeterministic=True
)

register(
    id='Woods101demi-v0',
    entry_point='gym_woods.envs:Woods101demi',
    max_episode_steps=500,
    nondeterministic=True
)

register(
    id='Woods102-v0',
    entry_point='gym_woods.envs:Woods102',
    max_episode_steps=500,
    nondeterministic=True
)

register(
    id='MazeF1-v0',
    entry_point='gym_maze.envs:MazeF1',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeF2-v0',
    entry_point='gym_maze.envs:MazeF2',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeF3-v0',
    entry_point='gym_maze.envs:MazeF3',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeF4-v0',
    entry_point='gym_maze.envs:MazeF4',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeF8-v0',
    entry_point='gym_maze.envs:MazeF8',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeF9-v0',
    entry_point='gym_maze.envs:MazeF9',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeH1-v0',
    entry_point='gym_maze.envs:MazeH1',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='Maze10-v0',
    entry_point='gym_maze.envs:Maze10',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeA-v0',
    entry_point='gym_maze.envs:MazeA',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeA_1-v0',
    entry_point='gym_maze.envs:MazeA_1',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeB-v0',
    entry_point='gym_maze.envs:MazeB',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeD-v0',
    entry_point='gym_maze.envs:MazeD',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeE1-v0',
    entry_point='gym_maze.envs:MazeE1',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeE2-v0',
    entry_point='gym_maze.envs:MazeE2',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeE3-v0',
    entry_point='gym_maze.envs:MazeE3',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='Maze4-v0',
    entry_point='gym_maze.envs:Maze4',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Maze5-v0',
    entry_point='gym_maze.envs:Maze5',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Maze6-v0',
    entry_point='gym_maze.envs:Maze6',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='Maze7-v0',
    entry_point='gym_maze.envs:Maze7',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeT2-v0',
    entry_point='gym_maze.envs:MazeT2',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeT3-v0',
    entry_point='gym_maze.envs:MazeT3',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeT4-v0',
    entry_point='gym_maze.envs:MazeT4',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='Maze228-v0',
    entry_point='gym_maze.envs:Maze228',
    max_episode_steps=250,
    nondeterministic=True
)

register(
    id='Maze252-v0',
    entry_point='gym_maze.envs:Maze252',
    max_episode_steps=250,
    nondeterministic=True
)

register(
    id='Maze288-v0',
    entry_point='gym_maze.envs:Maze288',
    max_episode_steps=250,
    nondeterministic=True
)

register(
    id='Maze324-v0',
    entry_point='gym_maze.envs:Maze324',
    max_episode_steps=250,
    nondeterministic=True
)

register(
    id='MazeH1-v0',
    entry_point='gym_maze.envs:MazeH1',
    max_episode_steps=80,
    nondeterministic=False
)