# Created by:
# markuz@kth.se 970328-T171
# amper@kth.se 971231-3817
import utils as ut
import time
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

maze = ut.Maze()
V, policy = ut.dynamic_programming(maze, 20)
path = maze.simulate((0, 0, 6, 5), policy)


class AnimateGame:
    def __init__(self, path):
        self.maze = ut.Maze()
        self.path = path

    def animate(self, i):
        moves = self.path[i]
        self.maze.board[moves[0]][moves[1]] = 2
        self.maze.board[moves[2]][moves[3]] = -1
        self.maze.draw()
        self.maze.board[moves[0]][moves[1]] = 0
        self.maze.board[moves[2]][moves[3]] = 0


total = AnimateGame(path)
fig = plt.figure(1, figsize=(total.maze.board.shape[1], total.maze.board.shape[1]))

anim = animation.FuncAnimation(fig, total.animate, frames=len(path), interval=300)
anim.save('results/mino.gif', writer='imagemagick')
