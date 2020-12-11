# CS747: Assignment 3

Author: Mithilesh Vaidya

# Directory Structure

* plots/: Generated plots stored in this directory.
* mdp.py: Contains an MDP class and the 3 algorithms: Sarsa, Expected Sarsa and Q-Learning.
* transitions.py: Contains functions which gives the next state in the grid and the reward, given the current state, action and wind.
* test.py: Script which runs experiments and generates required plots.
* report.pdf: Report summarises the findings.

# Usage

All plots are stored in the _/plots_ directory.

Run the command: "_python test.py --t x_" where x is:
* t2: run task 2 (simple 4 move agent). Generates a plot called 't2.png'.
* t3: run task 3 (8 move agent). Generates a plot called 't3.png'.
* t4: run task 4 (8 move + stochastic wind). Generates a plot called 't4.png'.
* t234: run task 2, 3 and 4 and plot the results on a single graph for easier comparison. A plot called 't234.png' is generated.
* t5: run task 5 (comparson of Sarsa, Expected Sarsa and Q-Learning). Generates a plot called 't5.png'.
* vary_a: Vary alpha as mentioned in the report and save a plot named 'varying_alpha.png'.
* vary_e: Vary epsilon as mentioned in the report and save a plot named 'varying_epsilon.png'.
* vary_r: Vary terminal reward as mentioned in the report and save a plot named 'varying_terminal_reward.png'.
* random_vs_sarsa: Plot performance of Sarsa(0) vs Random walk in case of 8 move + stochastic wind as mentioned in the report. Generates a plot named 'varying_terminal_reward.png'.

e.g. _python test.py --t t3_ : will run task 3
