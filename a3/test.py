from mdp import MDP
import numpy as np
import matplotlib.pyplot as plt
import argparse


rows, cols = 7, 10
wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

num_episodes = 200
num_trials = 15


def plot_one(res, title, savename):
    # y axis
    plt.clf()
    y = np.arange(res.shape[0])
    plt.plot(res, y)
    plt.xlabel("Time step")
    plt.ylabel("Episodes")
    plt.title(title)
    plt.grid(True)
    # plt.show()
    plt.savefig("plots/"+savename)


def task_2():    
    # simple 4 move
    res = np.zeros(num_episodes+1)
    a = MDP((rows, cols), (3, 0), (3, 7), wind, -1, 1, king_moves=False, stochastic_wind=False)
    epsilon, alpha = 0.1, 0.5    
    for seed in range(num_trials):        
        episodes = a.sarsa(seed, num_episodes, epsilon, alpha)
        res += np.cumsum(np.array(episodes))
    res /= num_trials
    plot_one(res, 'Sarsa(0), 4 move agent', 't2.png')

    return res

def task_3():
    # kings moves
    res = np.zeros(num_episodes+1)
    a = MDP((rows, cols), (3, 0), (3, 7), wind, -1, 1, king_moves=True, stochastic_wind=False)
    epsilon, alpha = 0.1, 0.5
    for seed in range(num_trials):
        episodes = a.sarsa(seed, num_episodes, epsilon, alpha)
        res += np.cumsum(np.array(episodes))
    res /= num_trials
    plot_one(res, 'Sarsa(0), 8 move agent', 't3.png')

    return res

def task_4():
    # stochastic wind
    res = np.zeros(num_episodes+1)
    a = MDP((rows, cols), (3, 0), (3, 7), wind, -1, 1, king_moves=True, stochastic_wind=True)
    epsilon, alpha = 0.1, 0.5
    for seed in range(num_trials):        
        episodes = a.sarsa(seed, num_episodes, epsilon, alpha)
        res += np.array(episodes)
    res = np.cumsum(res)/num_trials
    plot_one(res, 'Sarsa(0), 8 move agent, stochastic wind', 't4.png')

    return res

def plot_2_3_4(res2, res3, res4):
    # y axis
    plt.clf()
    y = np.arange(num_episodes+1)
    plt.plot(res2, y, label='4 moves')
    plt.plot(res3, y, label='8 moves')
    plt.plot(res4, y, label='8 moves+stochastic wind')
    plt.xlabel("Time step")
    plt.ylabel("Episodes")
    plt.title("Sarsa(0) agent")
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig("plots/t2_t3_t4.png")


def task_5():
    plt.clf()
    a = MDP((rows, cols), (3, 0), (3, 7), wind, -1, 1, king_moves=False, stochastic_wind=False)
    epsilon, alpha = 0.1, 0.5
    episodes_avg_t = np.zeros((3, num_episodes+1))

    for seed in range(num_trials):
        # sarsa
        episodes = a.sarsa(seed, num_episodes, epsilon, alpha)
        episodes_avg_t[0, :] += np.cumsum(np.array(episodes))
        # q-learning
        episodes = a.q_learning(seed, num_episodes, epsilon, alpha)
        episodes_avg_t[1, :] += np.cumsum(np.array(episodes))
        # expected sarsa
        episodes = a.expected_sarsa(seed, num_episodes, epsilon, alpha)
        episodes_avg_t[2, :] += np.cumsum(np.array(episodes))
    
    # normalise
    episodes_avg_t /= num_trials
    # y axis
    y = np.arange(num_episodes+1)
    plt.title("Comparison of various algos\n4 moves, no stochastic wind")
    plt.plot(episodes_avg_t[0], y, label='Sarsa')
    plt.plot(episodes_avg_t[1], y, label='Q-Learning')
    plt.plot(episodes_avg_t[2], y, label='Expected Sarsa')
    plt.xlabel("Time step")
    plt.ylabel("Episodes")
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig("plots/t5.png")

def changing_alpha():
    plt.clf()
    alpha_range = np.arange(0.1, 1, 0.1)
    # simple 4 move
    res = np.zeros((alpha_range.shape[0], num_episodes+1))
    a = MDP((rows, cols), (3, 0), (3, 7), wind, -1, 1, king_moves=False, stochastic_wind=False)
    epsilon = 0.1
    for i in range(alpha_range.shape[0]):
        alpha = alpha_range[i]
        # print("Testing for alpha = ", alpha)        
        for seed in range(num_trials):        
            episodes = a.sarsa(seed, num_episodes, epsilon, alpha)
            res[i, :] += np.cumsum(np.array(episodes))
        res[i] /= num_trials
        plt.plot(res[i], np.arange(num_episodes+1), label=str(round(alpha, 2)))
    
    plt.title("4 moves, varying alpha")
    plt.xlabel("Time step")
    plt.ylabel("Episodes")
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/varying_alpha.png')
    plt.show()

def changing_epsilon():
    plt.clf()
    epsilon_range = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    # simple 4 move
    res = np.zeros((epsilon_range.shape[0], num_episodes+1))
    a = MDP((rows, cols), (3, 0), (3, 7), wind, -1, 1, king_moves=False, stochastic_wind=False)
    alpha = 0.5
    for i in range(epsilon_range.shape[0]):
        epsilon = epsilon_range[i]
        # print("Testing for epsilon = ", epsilon)        
        for seed in range(num_trials):        
            episodes = a.sarsa(seed, num_episodes, epsilon, alpha)
            res[i, :] += np.cumsum(np.array(episodes))
        res[i] /= num_trials
        plt.plot(res[i], np.arange(num_episodes+1), label=str(epsilon))
    
    plt.title("Varying epsilon")
    plt.xlabel("Time step")
    plt.ylabel("Episodes")
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/varying_epsilon.png')
    plt.show()

def changing_terminal_reward():
    plt.clf()
    reward_range = np.array([0, 1, 1e2, 1e3, 1e4])
    # simple 4 move
    res = np.zeros((reward_range.shape[0], num_episodes+1))    
    alpha = 0.5
    epsilon = 0.1
    for i in range(reward_range.shape[0]):        
        reward = reward_range[i]
        a = MDP((rows, cols), (3, 0), (3, 7), wind, -1, reward, king_moves=False, stochastic_wind=False)
        # print("Testing for reward = ", reward)        
        for seed in range(num_trials):        
            episodes = a.sarsa(seed, num_episodes, epsilon, alpha)
            res[i, :] += np.cumsum(np.array(episodes))
        res[i] /= num_trials
        plt.plot(res[i], np.arange(num_episodes+1), label=str(reward))
    
    plt.title("Varying terminal reward")
    plt.xlabel("Time step")
    plt.ylabel("Episodes")
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/varying_terminal_reward.png')
    # plt.show()

def random_vs_sarsa():
    plt.clf()
    # sarsa
    sarsa = np.zeros(num_episodes+1)
    a = MDP((rows, cols), (3, 0), (3, 7), wind, -1, 1, king_moves=True, stochastic_wind=True)
    epsilon, alpha = 0.1, 0.5
    for seed in range(num_trials):        
        episodes = a.sarsa(seed, num_episodes, epsilon, alpha)
        sarsa += np.array(episodes)
    sarsa = np.cumsum(sarsa)/num_trials
    # random walk
    random_walk = np.zeros(num_episodes+1)
    for seed in range(num_trials):        
        episodes = a.random_walk(seed, num_episodes)
        random_walk += np.array(episodes)        
    random_walk = np.cumsum(random_walk)/num_trials

    plt.clf()
    y = np.arange(num_episodes+1)
    plt.plot(sarsa, y, label='sarsa')
    plt.plot(random_walk, y, label='random walk')    
    plt.xlabel("Time step")
    plt.ylabel("Episodes")
    plt.title("Sarsa(0) agent with 8 moves, stochastic wind")
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig("plots/random_walk.png")




if __name__ == "__main__":

    import os

    if not os.path.exists("plots/"):
        os.mkdir("plots")

    parser = argparse.ArgumentParser()
    parser.add_argument("--t",type=str)    
    args = parser.parse_args()
    task = args.t

    if task == 't2':
        _ = task_2()
    elif task == 't3':
        _ = task_3()
    elif task == 't4':
        _ = task_4()
    elif task == 't234':
        simple = task_2()
        king_moves = task_3()
        wind = task_4()
        plot_2_3_4(simple, king_moves, wind)
    elif task == 't5':
        task_5()
    elif task == 'vary_a':
        changing_alpha()
    elif task == 'vary_e':
        changing_epsilon()
    elif task == 'vary_r':
        changing_terminal_reward()
    elif task == 'random_vs_sarsa':
        random_vs_sarsa()


