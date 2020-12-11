import numpy as np
import random

def thompson(means, seed, horizon):
    # set seeds
    np.random.seed(seed)
    random.seed(seed)
    # set up arrays
    num_arms = len(means)
    f_s = np.zeros((num_arms, 2))
    actual_rewards = np.zeros(horizon)
    arm_pulls = np.zeros(num_arms)
    # actual algo
    for t in range(horizon):
        all_samples = []
        for arm in range(num_arms):
            all_samples.append(np.random.beta(f_s[arm, 1] + 1, f_s[arm, 0] + 1))
        
        arm_to_play = np.argmax(all_samples)
        arm_pulls[arm_to_play] += 1
        reward = np.random.binomial(1, means[arm_to_play])
        actual_rewards[t] = reward
        f_s[arm_to_play, reward] += 1
    
    return actual_rewards
    
    # max_e_reward = np.max(means)*horizon
    # actual_reward = np.sum(actual_rewards)

    # return max_e_reward - actual_reward


def thompson_hint(means, seed, horizon):    
    # set seeds
    np.random.seed(seed)
    random.seed(seed)
    # set up arrays
    num_arms = len(means)
    arm_distri = np.ones((num_arms, num_arms)) / num_arms
    mu_distri = np.ones((num_arms, num_arms)) / num_arms
    actual_rewards = np.zeros(horizon)
    # sort means and use them to build prior
    means_hint = np.sort(means)
    arm_pulls = np.zeros(num_arms)
    # actual algo
    for t in range(horizon):        
        all_arms_preds = np.zeros(num_arms)
        all_arms_probs = np.zeros(num_arms)
        all_arms_theta = np.zeros(num_arms)
        for arm in range(num_arms):
            best_prediction = np.random.choice(range(num_arms),p=arm_distri[arm])            
            all_arms_theta[arm] = best_prediction
            all_arms_preds[arm] = means_hint[best_prediction]
            all_arms_probs[arm] = arm_distri[arm][best_prediction]
        
        # original
        # arms_with_best_means = np.where(all_arms_preds == np.max(all_arms_preds))[0]
        # arms_probs = all_arms_probs[arms_with_best_means]
        # normalised_arms_probs = arms_probs/np.sum(arms_probs)
        # arm_to_play = np.random.choice(arms_with_best_means, p=normalised_arms_probs)        

        # new
        arms_with_best_means = np.where(all_arms_preds == np.max(all_arms_preds))[0]
        chosen_theta = all_arms_theta[arms_with_best_means[0]].astype(np.uint8)
        if arms_with_best_means.shape[0] == 1 or np.sum(mu_distri[chosen_theta]) == 0:
            arm_to_play = arms_with_best_means[0]            
        else:
            arm_to_play = np.random.choice(range(num_arms), p=mu_distri[chosen_theta])
        
        arm_pulls[arm_to_play] += 1
        reward = np.random.binomial(1, means[arm_to_play])
        actual_rewards[t] = reward
        # update arm beleif 
        update_num = (means_hint**reward) * ((1-means_hint)**(1-reward))
        arm_distri[arm_to_play] *= update_num
        arm_distri[arm_to_play] /= np.sum(arm_distri[arm_to_play])
        # update mu belief
        update_factor = arm_distri[:, chosen_theta]
        mu_distri[chosen_theta] *= update_factor
        if np.sum(mu_distri[chosen_theta]):
            mu_distri[chosen_theta] /= np.sum(mu_distri[chosen_theta])

    return actual_rewards
    # max_e_reward = np.max(means)*horizon
    # actual_reward = np.sum(actual_rewards)

    # return max_e_reward - actual_reward