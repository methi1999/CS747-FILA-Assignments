import numpy as np
import random

def ucb(means, seed, horizon):
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
        confidence_bound = np.sqrt(2*np.log(t)/num_pulls)
        ucb_val = cum_rewards/num_pulls + confidence_bound
        arm_to_play = np.argmax(ucb_val)
        reward = np.random.binomial(1, means[arm_to_play])
        actual_rewards[t] = reward
        cum_rewards[arm_to_play] += reward
        num_pulls[arm_to_play] += 1
    
    return actual_rewards

    # max_e_reward = np.max(means)*(horizon+num_arms)
    # actual_reward = np.sum(actual_rewards)

    # return max_e_reward - actual_reward

# print(ucb([0.2, 0.4, 0.6], 10, 12400))