import argparse
import numpy as np

from algos import epsilon_greedy, ucb, kl_ucb, thompson

# with open('../instances/i-1.txt', 'r') as f:
#     means = [float(x.strip()) for x in f.readlines()]

# sadha, hint = 0, 0
# total_runs = 50
# for seed in range(total_runs):
#     print(seed)
#     sadha = np.sum(thompson.thompson(means, seed, 102400))
#     hint = np.sum(thompson.thompson_hint(means, seed, 102400))
#     print('simple', sadha, 'hint', hint)
 
# print(np.sum(sadha)/total_runs, np.sum(hint)/total_runs)



def get_results(a, means, r, h, e):
    if a == 'epsilon-greedy':
        actual_rewards = epsilon_greedy.epsilon_greedy(means, r, h, e)
    elif a == 'ucb':
        actual_rewards = ucb.ucb(means, r, h)
    elif a == 'kl-ucb':
        actual_rewards = kl_ucb.kl_ucb(means, r, h)
    elif a == 'thompson-sampling':
        actual_rewards = thompson.thompson(means, r, h)
    elif a == 'thompson-sampling-with-hint':
        actual_rewards = thompson.thompson_hint(means, r, h)

    return actual_rewards

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS747 A1')
    parser.add_argument('--instance')
    parser.add_argument('--algorithm')
    parser.add_argument('--randomSeed')
    parser.add_argument('--epsilon')
    parser.add_argument('--horizon')
    parser.add_argument('--write_to_file', required=False) # by default false, but still mention

    args = parser.parse_args()
    i, a, r, e, h = args.instance, args.algorithm, int(args.randomSeed), float(args.epsilon), int(args.horizon)

    with open(i, 'r') as f:
        means = [float(x.strip()) for x in f.readlines()]

    if args.write_to_file is not None: # called by wrapper script        
        file_obj = open(args.write_to_file, 'a')
        horizons = [100, 400, 1600, 6400, 25600, 102400]
        # horizons = [100, 200, 1000]
        max_h = max(horizons)
        
        rewards = get_results(a, means, r, max_h, e)
        for horizon in horizons:            
            regret = np.max(means)*horizon - np.sum(rewards[:horizon])            
            final_out = ', '.join([i, a, str(r), str(e), str(horizon), str(regret)]) + '\n'
            file_obj.write(final_out)

    else: # simple call        
        rewards = get_results(a, means, r, h, e)
        regret = np.max(means)*h - np.sum(rewards)
        final_out = ', '.join([i, a, str(r), str(e), str(h), str(regret)]) + '\n'
        print(final_out)