import sys
import gym
import gym_maze
from gym_maze.common import maze_renderer

# MazeA-v0
print("=" * 50)
print("MazeA-v0")
print("=" * 50)

env = gym.make('MazeA-v0')
env.reset()

# Wyświetl w konsoli
maze_renderer.render(sys.stdout, env.env.maze.matrix)

# Zapisz do pliku
with open('MazeA-v0.txt', 'w', encoding='utf-8') as f:
    maze_renderer.render(f, env.env.maze.matrix)
print("Zapisano do MazeA-v0.txt")

env.close()

# MazeA_1-v0
print("\n" + "=" * 50)
print("MazeA_1-v0")
print("=" * 50)

env = gym.make('MazeA_1-v0')
env.reset()

# Wyświetl w konsoli
maze_renderer.render(sys.stdout, env.env.maze.matrix)

# Zapisz do pliku
with open('MazeA_1-v0.txt', 'w', encoding='utf-8') as f:
    maze_renderer.render(f, env.env.maze.matrix)
print("Zapisano do MazeA_1-v0.txt")

env.close()
