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
    # Actions
    # STAY       = 0
    # MOVE_LEFT  = 1
    # MOVE_RIGHT = 2
    # MOVE_UP    = 3
    # MOVE_DOWN  = 4

    # Reward values
    STEP_REWARD = 0
    EATEN_REWARD = -100
    WIN_REWARD = 1
    IMPOSSIBLE_REWARD = -100

    def __init__(self):
        self.board = np.zeros((7, 8))
        self.walls = [[0, 2], [1, 2], [2, 2], [3, 2], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [6, 4], [1, 5],
                      [2, 5], [3, 5], [2, 5], [2, 6], [2, 7]]
        self.colors = {0: 'WHITE', 1: 'BLACK', 2: 'GREEN', -1: 'RED'}
        self.states, self.eaten_states, self.win_states, self.map_ = self.__states();
        self.n_states = len(self.states);
        self.actions = self.__actions()
        self.n_actions = len(self.actions)
        self.transition_probabilities = self.__transitions();
        self.rewards = self.__rewards();
        self.minotaur = Minotaur()
        self.minotaur.generate_valid_moves(self.board)

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

    def __actions(self):
        actions = dict()
        actions[0] = (0, 0)
        actions[1] = (0, -1)
        actions[2] = (0, 1)
        actions[3] = (-1, 0)
        actions[4] = (1, 0)
        return actions

    def __states(self):
        states = {}
        eaten_states = []
        win_states = []
        map_ = {}
        s = 0
        for Py in range(self.board.shape[0]):
            for Px in range(self.board.shape[1]):
                for My in range(self.board.shape[0]):
                    for Mx in range(self.board.shape[1]):
                        if self.board[My, Mx] != 1 and self.board[Py, Px] != 1:
                            if Py == 6 and Px == 5:
                                win_states.append(s)

                            elif Py == My and Px == Mx:
                                eaten_states.append(s)

                            map_[(Py, Px, My, Mx)] = s
                            states[s] = (Py, Px, My, Mx)
                            s += 1

        return states, eaten_states, win_states, map_

    def __move(self, state, action):
        """
            State is a hash value.
            Action is an integer
            returns list of hashed next states given s and a
        """
        # Compute the future position given current (state, action)
        # The minotaur stops moving, either to eat your body,
        # or because you have escaped and he is hungry and wants to rest.

        # doesn't matter since the game ends if you die or ends if you win
        if state in self.eaten_states or self.win_states:
            return [state]

        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]

        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.board.shape[0]) or \
                             (col == -1) or (col == self.board.shape[1]) or \
                             (self.board[row, col] == 1)

        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            row = self.states[state][0]
            col = self.states[state][1]

        mino_moves = self.minotaur.valid_moves[self.states[state][-2:]]

        next_states = []

        # Create all possible minotaur positions from state s.
        for move in mino_moves:
            m_row = self.states[state][2] + self.actions[move][0]
            m_col = self.states[state][3] + self.actions[move][1]

            # jumping over wall
            if self.board[m_row, m_col] == 1:
                m_row += self.actions[move][0]
                m_col += self.actions[move][1]

            next_states.append(self.map_[(col, row, m_col, m_row)])

        return next_states

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):

            for a in range(self.n_actions):

                # List of possible next states from state s, given action a.
                next_states = self.__move(s, a)

                prob_weighted_reward = 0.0

                for next_s in next_states:

                    # Handle already been eaten:
                    if s in self.eaten_states:
                        prob_weighted_reward += self.STEP_REWARD

                    # Handle already have won:
                    elif s in self.win_states:
                        prob_weighted_reward += self.STEP_REWARD

                    # Handle being eaten:
                    elif next_s in self.eaten_states:
                        prob_weighted_reward += self.EATEN_REWARD

                    # Handle winning:
                    elif next_s in self.win_states:
                        prob_weighted_reward += self.WIN_REWARD

                    # Reward for hitting a wall
                    elif s == next_s and a != 0:
                        prob_weighted_reward += self.IMPOSSIBLE_REWARD

                        # Reward for taking a step to an empty cell that is not the exit
                    else:
                        prob_weighted_reward += self.STEP_REWARD

                rewards[s, a] = prob_weighted_reward / len(next_states)

        return rewards

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s, a)

                # if already eaten or won, then prob == 1
                prob = 1 / len(next_s)

                for new in next_s:
                    transition_probabilities[new, s, a] = prob

        return transition_probabilities


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1))
    policy = np.zeros((n_states, T + 1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)

    return V, policy


def simulate(self, start, policy):
    path = list();

    # Deduce the horizon from the policy shape
    horizon = policy.shape[1]
    # Initialize current state and time
    t = 0
    s = self.map[start]
    # Add the starting position in the maze to the path
    path.append(start)
    while t < horizon - 1:
        # Move to next state given the policy and the current state
        next_s = self.__move(s, policy[s, t])
        # Add the position in the maze corresponding to the next state
        # to the path
        path.append(self.states[next_s])
        # Update time and state for next iteration
        t += 1
        s = next_s
