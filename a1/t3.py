"""
Find epsilon values for task 3
"""

from algos import epsilon_greedy
import numpy as np
import matplotlib.pyplot as plt

instance_l = ['../instances/i-'+str(i+1)+'.txt' for i in range(3)]

for inst_num, instance in enumerate(instance_l):
    
    print(instance)
    
    with open(instance, 'r') as f:
        means = [float(x.strip()) for x in f.readlines()]

    num_arms = len(means)
    horizon = 102400
    seed_range = range(50)
    # e_range = [1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    e_range = [5e-3, 7.5e-3, 0.01, 0.015, 0.02, 0.025, 0.03]
    regrets = []

    for e in e_range:
        reward = 0
        for seed in seed_range:
            reward += np.sum(epsilon_greedy.epsilon_greedy(means, seed, horizon, e))
        regret = np.max(means)*horizon - reward/len(seed_range)
        regrets.append(regret)

    plt.plot(e_range, regrets, label="Instance:"+str(inst_num+1))

plt.xlabel("epsilon")
plt.ylabel("regret")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.show()
