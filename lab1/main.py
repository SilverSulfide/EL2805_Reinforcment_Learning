# Created by:
# markuz@kth.se 970328-T171
# amper@kth.se 971231-3817

import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt


def maze_main(args):
    import maze as ut
    maze = ut.Maze(stay=args.stay, jump=args.jump)

    V, policy = ut.dynamic_programming(maze, 18)
    path = maze.simulate((0, 0, 6, 5), policy, method='DynProg')

    # init animation
    total = ut.AnimateGame(path)
    fig = plt.figure(1, figsize=(total.maze.board.shape[1], total.maze.board.shape[1]))

    anim = animation.FuncAnimation(fig, total.animate, frames=len(path), interval=1500)
    anim.save('results/mino.gif', writer='imagemagick')

    # ------------- Dynamic Programming survival --------------
    #ut.survival_rate_dynprog(maze)

    # ------------- Dynamic Programming survival with staying --------------
    #maze2 = ut.Maze(stay=True)
    #ut.survival_rate_dynprog(maze2)

    # ------------- Value iteration --------------
    #ut.survival_rate_valiter(maze)


def city_main(args):
    import city as ut
    # init the main class
    city = ut.City()

    # Mean lifetime

    # Discount factor
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Vs = []

    for gamma in gammas:
        print(gamma)
        # Threshold
        epsilon = 0.0001

        V, policy = ut.value_iteration(city, gamma, epsilon)

        # get the starting state
        idx = city.map_[city.start_state]
        Vs.append(V[idx])

        city.draw(policy, plot=True, arrows=True)

    plt.plot(gammas, Vs, color='blue')
    plt.xlabel("Discount factor")
    plt.ylabel("Expected starting reward")
    plt.scatter(gammas, Vs, color='blue')
    plt.show()
    path = city.simulate(city.start_state, policy, method="ValIter", survival_factor=gamma)

    # init animation
    total = ut.AnimateGame(path)
    fig = plt.figure(1, figsize=(total.city.board.shape[1], total.city.board.shape[1]))

    anim = animation.FuncAnimation(fig, total.animate, frames=len(path), interval=300)
    anim.save('results/bank.gif', writer='imagemagick')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--problem', type=str, help='Problem name: maze or city')
    parser.add_argument('--stay', action='store_true', help='Enable minotaur staying')
    parser.add_argument('--jump', action='store_true', help='Enable minotaur jumping')

    args = parser.parse_args()
    print("Args: ", args)

    if args.problem == 'city':
        city_main(args)
    elif args.problem == 'maze':
        maze_main(args)
    else:
        sys.exit("Incorrect problem input")
