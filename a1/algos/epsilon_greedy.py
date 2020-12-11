import numpy as np
import random

def epsilon_greedy(means, seed, horizon, epsilon):
    # set seeds
    np.random.seed(seed)
    random.seed(seed)
    # set up arrays
    num_arms = len(means)
    cum_rewards = np.zeros(num_arms)
    num_pulls = np.zeros(num_arms)
    actual_rewards = np.zeros(horizon)
    # 1 round robin stage. Mention in report
    for i in range(min(num_arms, horizon)):
        num_pulls[i] += 1
        reward = np.random.binomial(1, means[i])
        cum_rewards[i] = reward
        actual_rewards[i] = reward
    # actual algo
    for t in range(min(num_arms, horizon), horizon):
        if (np.random.rand() <= epsilon):
            arm_to_play = np.random.randint(num_arms)
        else:
            empirical_means = cum_rewards/num_pulls
            arm_to_play = np.argmax(empirical_means)

        reward = np.random.binomial(1, means[arm_to_play])
        actual_rewards[t] = reward
        cum_rewards[arm_to_play] += reward
        num_pulls[arm_to_play] += 1
    
    return actual_rewards

    # max_e_reward = np.max(means)*(horizon+num_arms)
    # actual_reward = np.sum(actual_rewards)

    # return max_e_reward - actual_reward


# print(epsilon_greedy([0.2, 0.4, 0.6], 15, 1000, 0.5))