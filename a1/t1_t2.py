import bandit
import os

instance_l = ['../instances/i-'+str(i+1)+'.txt' for i in range(3)]
seeds = range(50)
# seeds = range(20)

task = 2

if task == 1:
    algo_names = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']
    write_to = 'outputDataT1.txt'
else:
    algo_names = ['thompson-sampling', 'thompson-sampling-with-hint']
    write_to = 'outputDataT2.txt'


for instance in instance_l:
    print("Instance:", instance)
    for algo in algo_names:
        print("Algo:", algo)
        for seed in seeds:            
            print(seed)
            cmd = "python bandit.py --instance \'{}' --algorithm {} --randomSeed {} --epsilon 0.02 --horizon 0 --write_to_file {}".format(instance, algo, seed, write_to)
            os.system(cmd)


# remove last new line character
all_chars = open(write_to, 'r').read()
open(write_to, 'w').write(all_chars[:-1])