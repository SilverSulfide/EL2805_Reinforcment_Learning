# Created by:
# markuz@kth.se 970328-T171
# amper@kth.se 971231-3817
import utils as ut
import time

maze = ut.Maze()
maze.init_board()

"""
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
"""

V, policy = ut.dynamic_programming(maze, 20)

"""
start  = (0,0,6,5);
path = maze.simulate(start, policy)

# Show the shortest path 
ut.animate_solution(maze, path)
"""