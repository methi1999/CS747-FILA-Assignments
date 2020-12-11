import numpy as np
from mdp import MDP
import pulp

def value_iteration(mdp):
    
    eps = 1e-10
    diff = np.inf
    value_func = np.random.rand(mdp.num_states)

    while diff > eps:
        prod = np.multiply(mdp.t, mdp.r)
        t1 = np.sum(prod, axis=2)
        t2 = mdp.gamma * np.sum(np.multiply(mdp.t, value_func[None, None, :]), axis=2)
        final = np.max(t1+t2, axis=1)
        diff = np.sum(np.abs(final - value_func))
        value_func = final
    
    mdp.value_function = value_func
    mdp.update_action_value()
    mdp.policy = np.argmax(mdp.action_value, axis=1)


def howard_pi(mdp):
    mdp.update_value_function()
    mdp.update_action_value()
    
    eps = 1e-10
    while 1:
        improvable = False
        for state in range(mdp.num_states):
            ia = mdp.action_value[state] > mdp.value_function[state] + eps
            ia[mdp.policy[state]] = False
            # action value and value function may not match precisely due to FP errors
            if np.any(ia):
                improvable = True
                mdp.policy[state] = np.where(ia == True)[0][0]

        mdp.update_value_function()
        mdp.update_action_value()
        if not improvable:
            break


def lp(mdp):
    model = pulp.LpProblem("MDP_Solver", pulp.LpMinimize)
    # define variables
    state_dict = pulp.LpVariable.dicts("states",
                                       (i for i in range(mdp.num_states)),
                                       cat='Continuous')
    # objective
    model += pulp.lpSum([state_dict[i] for i in range(mdp.num_states)])
    # set up constraints
    for state in range(mdp.num_states):
        for action in range(mdp.num_actions):
            model += pulp.lpSum([mdp.t[state, action, i]*(mdp.gamma*state_dict[i] + mdp.r[state, action, i]) for i in range(mdp.num_states)]) <= state_dict[state]
    # solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    # get the optimal value function
    for state in state_dict:
        mdp.value_function[state] = state_dict[state].varValue
    mdp.update_action_value()
    mdp.policy = np.argmax(mdp.action_value, axis=1)
    mdp.update_value_function()
    

if __name__ == "__main__":    
    a = MDP('/Users/mithileshvaidya/Documents/Sem 7/FILA/a2/data/mdp/continuing-mdp-50-20.txt')
    # value_iteration(a)
    # howard_pi(a)
    # lp(a)
    print(a.prettyPrint())