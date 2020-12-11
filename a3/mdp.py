import numpy as np
import random
import transitions


class MDP:

    def __init__(self, grid_dim, start_state, end_state, column_wind, non_terminal_reward, terminal_reward, king_moves, stochastic_wind):

        (self.rows, self.cols) = grid_dim
        (self.start_x, self.start_y) = start_state
        (self.end_x, self.end_y) = end_state

        self.num_states = self.rows*self.cols
        self.gamma = 1
        self.stochastic_wind = stochastic_wind

        # for 4: top, right, bottom, left
        # for 8: top, top_right, ....
        if king_moves:
            self.num_actions = 8
            self.next_state_func = transitions.get_next_state_8
        else:
            self.num_actions = 4
            self.next_state_func = transitions.get_next_state_4

        self.start = self.start_x*self.cols + self.start_y
        self.end = self.end_x*self.cols + self.end_y

        self.column_wind = column_wind
        self.terminal_reward = terminal_reward
        self.non_terminal_reward = non_terminal_reward

    def reset_action_value(self):
        self.action_value = np.zeros((self.num_states, self.num_actions))

    def sarsa(self, seed, num_episodes, epsilon, alpha):
        self.reset_action_value()
        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        # store episode times
        episode_times = [0]
        for _ in range(num_episodes):
            t = 0
            cur_state = self.start
            cur_action = np.random.randint(low=0, high=self.num_actions)
            while cur_state != self.end:
                # next state
                next_state, reward = self.next_state_func(
                    self.rows, self.cols, self.end_x, self.end_y,
                    self.non_terminal_reward, self.terminal_reward, self.column_wind, self.stochastic_wind,
                    cur_state//self.cols, cur_state % self.cols, cur_action)

                # get distribution over actions
                best_action = np.argmax(self.action_value[next_state])
                action_distri = np.ones(
                    self.num_actions)*epsilon/self.num_actions
                action_distri[best_action] += (1-epsilon)
                # choose action
                next_action = np.random.choice(
                    np.arange(self.num_actions), p=action_distri, size=1)[0]
                # update Qt
                target = self.action_value[next_state, next_action]
                self.action_value[cur_state, cur_action] += alpha*(
                    reward + self.gamma*target - self.action_value[cur_state, cur_action])
                # update state, action
                cur_state = next_state
                cur_action = next_action
                t += 1
            episode_times.append(t)        

        return episode_times

    def q_learning(self, seed, num_episodes, epsilon, alpha):
        self.reset_action_value()
        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        # store episode times
        episode_times = [0]
        for _ in range(num_episodes):
            t = 0
            cur_state = self.start
            while cur_state != self.end:
                # get distribution over actions
                best_action = np.argmax(self.action_value[cur_state])
                action_distri = np.ones(
                    self.num_actions)*epsilon/self.num_actions
                action_distri[best_action] += (1-epsilon)
                # choose action
                action = np.random.choice(
                    np.arange(self.num_actions), p=action_distri, size=1)[0]
                # next state
                next_state, reward = self.next_state_func(
                    self.rows, self.cols, self.end_x, self.end_y,
                    self.non_terminal_reward, self.terminal_reward, self.column_wind, self.stochastic_wind,
                    cur_state//self.cols, cur_state % self.cols, action)
                # update Qt
                target = np.max(self.action_value[next_state])
                self.action_value[cur_state, action] += alpha * \
                    (reward + self.gamma*target -
                     self.action_value[cur_state, action])
                # update state, action
                cur_state = next_state
                t += 1
            episode_times.append(t)

        return episode_times

    def expected_sarsa(self, seed, num_episodes, epsilon, alpha):
        self.reset_action_value()
        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        # store episode times
        episode_times = [0]
        for _ in range(num_episodes):
            t = 0
            cur_state = self.start
            cur_action = np.random.randint(low=0, high=self.num_actions)
            while cur_state != self.end:
                # next state
                next_state, reward = self.next_state_func(
                    self.rows, self.cols, self.end_x, self.end_y,
                    self.non_terminal_reward, self.terminal_reward, self.column_wind, self.stochastic_wind,
                    cur_state//self.cols, cur_state % self.cols, cur_action)

                # get distribution over actions
                best_action = np.argmax(self.action_value[next_state])
                action_distri = np.ones(
                    self.num_actions)*epsilon/self.num_actions
                action_distri[best_action] += (1-epsilon)
                # choose action
                next_action = np.random.choice(
                    np.arange(self.num_actions), p=action_distri, size=1)[0]
                # update Qt
                target = np.dot(self.action_value[next_state], action_distri)
                self.action_value[cur_state, cur_action] += alpha*(
                    reward + self.gamma*target - self.action_value[cur_state, cur_action])
                # update state, action
                cur_state = next_state
                cur_action = next_action
                t += 1
            episode_times.append(t)

        return episode_times

    def random_walk(self, seed, num_episodes):
        self.reset_action_value()
        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        # store episode times
        episode_times = [0]
        for _ in range(num_episodes):
            t = 0
            cur_state = self.start
            while cur_state != self.end:                
                cur_action = np.random.randint(self.num_actions)

                next_state, _ = self.next_state_func(
                    self.rows, self.cols, self.end_x, self.end_y,
                    self.non_terminal_reward, self.terminal_reward, self.column_wind, self.stochastic_wind,
                    cur_state//self.cols, cur_state % self.cols, cur_action)                
                
                cur_state = next_state
                t += 1
                
            episode_times.append(t)

        return episode_times

    def get_greeedy_path(self):

        # greedy path
        greedy_actions = []
        cur_state = self.start
        while cur_state != self.end:
            # choose action
            a = np.argmax(self.action_value[cur_state])
            greedy_actions.append(a)
            next_s, _ = self.next_state_func(cur_state//self.cols, cur_state%self.cols, a)
            cur_state = next_s
        
        return greedy_actions