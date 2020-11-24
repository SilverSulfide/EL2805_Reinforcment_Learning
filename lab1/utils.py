import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class Minotaur:
    def __init__(self, stay=False):
        self.x = 5
        self.y = 6
        # stay, left, right, up, down : (y,x)
        if stay:
            self.actions = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        else:
            self.actions = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        self.action_names = {0: 'stay', 1: 'left', 2: 'right', 3: 'up', 4: 'down'}
        self.valid_moves = defaultdict(list)

    def move(self, board):

        moves = self.valid_moves[(self.y, self.x)]
        move = np.random.choice(moves)

        self.y += self.actions[move][0]
        self.x += self.actions[move][1]

        if board[self.y, self.x] == 1:
            self.y += self.actions[move][0]
            self.x += self.actions[move][1]

    def generate_valid_moves(self, board):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                # doing the 0 move
                for move in self.actions:
                    if board[i][j] != 1:
                        temp_y = i + self.actions[move][0]
                        temp_x = j + self.actions[move][1]

                        if temp_y != -1 and temp_x != -1:

                            try:
                                location = board[temp_y][temp_x]
                                if location == 1:
                                    temp_y += self.actions[move][0]
                                    temp_x += self.actions[move][1]

                                    try:
                                        location = board[temp_y][temp_x]
                                        if location == 0:
                                            self.valid_moves[(i, j)].append(move)
                                    except:
                                        dummy = 0
                                else:
                                    self.valid_moves[(i, j)].append(move)

                            except:
                                dummy = 0


class Player:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.actions = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        self.action_names = {0: 'stay', 1: 'left', 2: 'right', 3: 'up', 4: 'down'}

    def move(self, board, move):
        # Compute the future position given current (state, action)

        self.y += self.actions[move][0]
        self.x += self.actions[move][1]

        # Is the future position an impossible one ?
        hitting_maze_walls = (self.y == -1) or (self.y == board.shape[0]) or \
                             (self.x == -1) or (self.x == board.shape[1]) or \
                             (board[self.y, self.x] == 1)

        # Based on the impossibilty check return the next state.
        if hitting_maze_walls:
            self.y -= self.actions[move][0]
            self.x -= self.actions[move][1]


class Maze:
    def __init__(self):
        self.board = np.zeros((7, 8))
        self.walls = [[0, 2], [1, 2], [2, 2], [3, 2], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [6, 4], [1, 5],
                      [2, 5], [3, 5], [2, 5], [2, 6], [2, 7]]
        self.colors = {0: 'WHITE', 1: 'BLACK', 2: 'GREEN', -1: 'RED'}

    def init_board(self):
        for wall in self.walls:
            self.board[wall[0]][wall[1]] = 1

    def draw(self):
        # Give a color to each cell
        rows, cols = self.board.shape
        colored_maze = [[self.colors[self.board[j, i]] for i in range(cols)] for j in range(rows)]

        # Draw start and end
        # FIXME: hardcoded
        text_maze = [['A', None, None, None, None, None, None, None]]
        for i in range(5):
            text_maze.append([None, None, None, None, None, None, None, None])
        text_maze.append([None, None, None, None, None, 'B', None, None])

        # Create figure of the size of the maze
        fig = plt.figure(1, figsize=(cols, rows))

        # Remove the axis ticks and add title
        ax = plt.gca()
        ax.set_title('The Maze')
        ax.set_xticks([])
        ax.set_yticks([])

        # Create a table to color
        grid = plt.table(cellText=text_maze,
                         cellColours=colored_maze,
                         cellLoc='center',
                         loc=(0, 0),
                         edges='closed')

        # Modify the height and width of the cells in the table
        tc = grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0 / rows)
            cell.set_width(1.0 / cols)

        plt.show(block=False)
        plt.pause(0.1)
        plt.close("all")
