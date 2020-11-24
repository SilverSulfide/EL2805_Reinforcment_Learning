# Created by:
# markuz@kth.se 970328-T171
# amper@kth.se XXXXXX
import utils as ut
from IPython import display
import time

maze = ut.Maze()
maze.init_board()

mino = ut.Minotaur()
maze.board[mino.y][mino.x] = -1
maze.draw()
maze.board[mino.y][mino.x] = 0
mino.generate_valid_moves(maze.board)

while True:
    mino.move(maze.board)
    maze.board[mino.y][mino.x] = -1
    maze.draw()
    maze.board[mino.y][mino.x] = 0