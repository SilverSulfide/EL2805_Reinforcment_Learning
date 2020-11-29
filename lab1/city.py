import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class Police:
    def __init__(self):
        self.x = 2
        self.y = 1
        # left, right, up, down : (y,x)
        self.actions = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        self.action_names = {1: 'left', 2: 'right', 3: 'up', 4: 'down'}
        self.valid_moves = defaultdict(list)

    def move(self, player_y, player_x):

        # grab the available moves
        moves = [1, 2, 3, 4]

        # moving logic
        # check same column
        if player_x == self.x:
            # check if we are below of the player
            if player_y < self.y:
                moves.remove(4)
            # check if we are above the player
            elif self.y < player_y:
                moves.remove(3)

        # check the same line
        elif player_y == self.y:
            # check if we are on the left of the player
            if player_x > self.x:
                moves.remove(1)
            # check if we are on the right of the player
            elif self.x > player_x:
                moves.remove(2)

        else:
            # we are below on the left
            if player_y < self.y and player_x > self.x:
                moves.remove(4)
                moves.remove(1)

            # we are above the left
            if player_y > self.y and player_x > self.x:
                moves.remove(3)
                moves.remove(1)

            # we are below on the right
            if player_y < self.y and player_x < self.x:
                moves.remove(4)
                moves.remove(2)

            # we are above on the right
            if player_y > self.y and player_x < self.x:
                moves.remove(3)
                moves.remove(2)

        # grab valid moves
        valid_moves = set(self.valid_moves[(self.y, self.x)])
        # intersect valid moves with chasing moves
        moves = list(valid_moves.intersection(set(moves)))

        return moves

    def generate_valid_moves(self, board):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                for move in self.actions:
                    temp_y = i + self.actions[move][0]
                    temp_x = j + self.actions[move][1]

                    if temp_y != -1 and temp_x != -1:

                        try:
                            location = board[temp_y][temp_x]
                            self.valid_moves[(i, j)].append(move)

                        except:
                            dummy = 0


class City:

    def __init__(self):
        # reward values
        self.STEP_REWARD = 0
        self.BANK_REWARD = 10
        self.CAUGHT_REWARD = -50
        self.IMPOSSIBLE_REWARD = -10000000000

        self.banks = [[0, 0], [2, 0], [0, 5], [2, 5]]
        self.police_station = [1, 2]

        self.board = self.__init_board()
        self.police = Police()
        self.police.generate_valid_moves(self.board)

        self.start_state = (0, 0, 1, 2)

        self.colors = {0: 'WHITE', 1: 'BLUE', 2: 'GREEN', -1: 'RED'}
        self.action_names = {0: 'stay', 1: 'left', 2: 'right', 3: 'up', 4: 'down'}
        self.action_arrows = {0: ".", 1: '←', 2: '→', 3: '↑', 4: '↓'}

        self.states, self.caught_states, self.bank_states, self.map_ = self.__states()
        self.n_states = len(self.states)

        self.actions = self.__actions()
        self.n_actions = len(self.actions)

        self.transition_probabilities = self.__transitions()

        self.rewards = self.__rewards()

    def __init_board(self):
        board = np.zeros((3, 6))
        for bank in self.banks:
            board[bank[0]][bank[1]] = 1
        board[self.police_station[0]][self.police_station[1]] = -1
        print(board)
        return board

    def draw(self, policy, plot=False, arrows=False):
        # Give a color to each cell
        rows, cols = self.board.shape
        colored_maze = [[self.colors[self.board[j, i]] for i in range(cols)] for j in range(rows)]

        # Draw start and end
        text_maze = [['Bank1', None, None, None, None, 'Bank4'], [None, None, None, None, None, None],
                     ['Bank3', None, None, None, None, 'Bank5']]

        # Create figure of the size of the maze
        if plot:
            fig = plt.figure(1, figsize=(cols, rows))

        # Remove the axis ticks and add title
        ax = plt.gca()
        ax.clear()
        ax.set_title('The City')
        ax.set_xticks([])
        ax.set_yticks([])

        if arrows:
            for Ry in range(self.board.shape[0]):
                for Rx in range(self.board.shape[1]):
                    if Ry != 1 or Rx != 2:
                        s = self.map_[(Ry, Rx, 1, 2)]
                        text_maze[Ry][Rx] = self.action_arrows[policy[s]]

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

        if plot:
            plt.show()

    def __actions(self):
        actions = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        return actions

    def __states(self):
        states = {}
        caught_states = []
        bank_states = []
        map_ = {}
        s = 0
        for Ry in range(self.board.shape[0]):
            for Rx in range(self.board.shape[1]):
                for Py in range(self.board.shape[0]):
                    for Px in range(self.board.shape[1]):

                        # check if caught by da police
                        if Ry == Py and Rx == Px:
                            caught_states.append(s)

                        # we wuz kangz n shieeeet
                        # https://www.youtube.com/watch?v=ofjZ2HY_uh8&ab_channel=AkkadDaily
                        elif [Ry, Rx] in self.banks:
                            bank_states.append(s)

                        map_[(Ry, Rx, Py, Px)] = s
                        states[s] = (Ry, Rx, Py, Px)
                        s += 1

        return states, caught_states, bank_states, map_

    def __move(self, state, action):
        """
            State is a hash value.
            Action is an integer
            returns list of hashed next states given s and a
        """

        if state in self.caught_states:
            return [self.map_[self.start_state]]

        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]

        # Is the future position an impossible one ?
        hitting_city_walls = (row == -1) or (row == self.board.shape[0]) or \
                             (col == -1) or (col == self.board.shape[1])

        # Based on the impossiblity check return the next state.
        if hitting_city_walls:
            row = self.states[state][0]
            col = self.states[state][1]

        # grab copper moves
        self.police.y = self.states[state][2]
        self.police.x = self.states[state][3]
        cops_moves = self.police.move(self.states[state][0], self.states[state][1])
        next_states = []

        # Create all possible cop positions from state s.
        for move in cops_moves:
            m_row = self.states[state][2] + self.actions[move][0]
            m_col = self.states[state][3] + self.actions[move][1]

            next_states.append(self.map_[(row, col, m_row, m_col)])

        return next_states

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):

            for a in range(self.n_actions):

                # List of possible next states from state s, given action a.
                next_states = self.__move(s, a)

                # we don't know where the police will go, so we need to add a weight to the reward,
                # because the action has a probability to move to different states, with different rewards.
                prob_weighted_reward = 0.0

                for next_s in next_states:

                    # Handle getting caught:
                    if next_s in self.caught_states:
                        prob_weighted_reward += self.CAUGHT_REWARD

                    # handle bank state:
                    elif next_s in self.bank_states:
                        prob_weighted_reward += self.BANK_REWARD

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
        # Initialize the transition probabilities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s, a)

                # if already eaten or won, then prob == 1
                prob = 1.0 / len(next_s)

                for new in next_s:
                    transition_probabilities[new, s, a] = prob

        return transition_probabilities

    def simulate(self, start, policy, method, survival_factor=None):
        path = []

        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 0
            s = self.map_[start]

            # Loop while state is not the goal state
            while True:

                path.append(self.states[s])

                # Move to next state given the policy and the current state
                if s in self.caught_states:
                    break

                next_s = self.__move(s, policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                next_s = np.random.choice(next_s)

                # Update time and state for next iteration
                t += 1
                s = next_s
                if t > 100:
                    break

        return path


# Animator class
class AnimateGame:
    def __init__(self, path):
        self.city = City()
        self.path = path

    def animate(self, i):
        moves = self.path[i]
        self.city.board[moves[0]][moves[1]] = 2
        self.city.board[moves[2]][moves[3]] = -1
        self.city.draw(None, plot=False, arrows=False)

        self.city.board[moves[0]][moves[1]] = 0
        self.city.board[moves[2]][moves[3]] = 0

        # reset bank colors
        for bank in self.city.banks:
            self.city.board[bank[0]][bank[1]] = 1


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))

    # we start at bank1 so we have reward 10
    Q[:, 0] = 10
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    if gamma > 0:
        tol = (1 - gamma) * epsilon / gamma
    else:
        tol = 0

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 10000:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        # print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy
