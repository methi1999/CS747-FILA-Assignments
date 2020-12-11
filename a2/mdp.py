import numpy as np
import random

class MDP:

    def __init__(self, filepath):

        with open(filepath, 'r') as f:
            inp = [x.strip() for x in f.readlines()]
        
        self.num_states = int(inp[0].split(' ')[-1])
        self.num_actions = int(inp[1].split(' ')[-1])
        self.t = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.r = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.start = int(inp[2].split(' ')[-1])
        self.end_states = [int(x) for x in inp[3].split(' ')[1:]]
        
        # read transition probs and rewards
        i = 4
        while inp[i].split(' ')[0] == 'transition':
            s1, a, s2, r, p = inp[i].split(' ')[1:]
            s1, a, s2 = int(s1), int(a), int(s2)
            r, p = float(r), float(p)
            if s1 not in self.end_states:
                self.t[s1, a, s2] = p
                self.r[s1, a, s2] = r
            i += 1
        
        self.mdptype = inp[i].split(' ')[-1]
        self.gamma = float(inp[i+1].split(' ')[-1])

        self.policy = np.random.randint(0, self.num_actions, size=self.num_states)
        self.value_function = np.zeros(self.num_states)
        self.action_value = np.zeros((self.num_states, self.num_actions))

    def update_value_function(self):
        # Ax = b
        a = np.eye(self.num_states)
        b = np.zeros(self.num_states)
        for state in range(self.num_states):
            a[state] -= self.gamma*self.t[state, self.policy[state]]
            b[state] = np.dot(self.r[state, self.policy[state]], self.t[state, self.policy[state]])

        self.value_function = np.matmul(np.linalg.inv(a), b)

    def update_action_value(self):
        t1 = np.sum(np.multiply(self.t, self.r), axis=2)
        t2 = self.gamma*np.sum(np.multiply(self.t, self.value_function[None, None, :]), axis=2)
        self.action_value = t1 + t2

    def __str__(self):

        des = ''
        des += '# states: {}, # actions: {}\n'.format(str(self.num_states), str(self.num_actions))
        des += 'Type: {}\n'.format(self.mdptype)
        des += 'Gamma: {}\n'.format(str(self.gamma))
        des += 'T: {}, R: {}\n'.format(self.t, self.r)
        return des

    def prettyPrint(self):
        s = ''
        for state in range(self.num_states):
            s += '{0:.6f}'.format(self.value_function[state]) + ' {}\n'.format(self.policy[state])
        return s[:-1]



# a = MDP('/Users/mithileshvaidya/Documents/Sem 7/FILA/a2/mdp.txt')
# print(a)
# a.update_value_function()