# Created by:
# markuz@kth.se 970328-T171
# amper@kth.se 971231-3817
import utils as ut
import time
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

maze = ut.Maze()
"""
V, policy = ut.dynamic_programming(maze, 20)
path = maze.simulate((0, 0, 6, 5), policy, method='DynProg')

# init animation
total = ut.AnimateGame(path)
fig = plt.figure(1, figsize=(total.maze.board.shape[1], total.maze.board.shape[1]))

anim = animation.FuncAnimation(fig, total.animate, frames=len(path), interval=300)
anim.save('results/mino.gif', writer='imagemagick')
"""

# ------------- Value iteration --------------
ut.survival_rate_valiter(maze)

