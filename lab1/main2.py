# Created by:
# markuz@kth.se 970328-T171
# amper@kth.se 971231-3817

# import utils as ut
import utils_rob as ut
import time
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

city = ut.City()

# Mean lifetime

# Discount factor
gammas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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

print(len(path))
print(path)

# init animation
total = ut.AnimateGame(path)
fig = plt.figure(1, figsize=(total.maze.board.shape[1], total.maze.board.shape[1]))

anim = animation.FuncAnimation(fig, total.animate, frames=len(path), interval=300)
anim.save('results/bank.gif', writer='imagemagick')
