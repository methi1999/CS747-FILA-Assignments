import numpy as np
import random

def kl_divergence(p, q):
    if q == 1:
        return np.inf
    elif q == p:
        return 0
    elif p == 0:
        return 0
    else:
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))


def find_q_recursive(l, r, num_pulls, empirical_mean, t, c, eps):
    q = (l+r)/2
    
    lhs = num_pulls*kl_divergence(empirical_mean, q)
    if lhs == np.inf:
        return q
    rhs = np.log(t) #+ max(0, c*np.log(np.log(t)))
    
    if abs(lhs-rhs) < eps:
        return q
    elif lhs < rhs:
        return find_q_recursive(q, r, num_pulls, empirical_mean, t, c, eps)
    else:
        return find_q_recursive(l, q, num_pulls, empirical_mean, t, c, eps)


def kl_ucb(means, seed, horizon):
    # set seeds
    np.random.seed(seed)
    random.seed(seed)
    # parameters c and eps
    c = 3
    eps = 1e-4
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
        kl_ucb_vals = []        
        for arm in range(num_arms):
            p_hat = cum_rewards[arm]/num_pulls[arm]
            kl_ucb_vals.append(find_q_recursive(p_hat, 1, num_pulls[arm], p_hat, t, c, eps))
        
        arm_to_play = np.argmax(kl_ucb_vals)
        reward = np.random.binomial(1, means[arm_to_play])
        actual_rewards[t] = reward
        cum_rewards[arm_to_play] += reward
        num_pulls[arm_to_play] += 1
    
    return actual_rewards

    # max_e_reward = np.max(means)*(horizon+num_arms)
    # actual_reward = np.sum(actual_rewards)
    
    # return max_e_reward - actual_reward

# print(kl_ucb([0.2, 0.4, 0.6], 7, 1000))