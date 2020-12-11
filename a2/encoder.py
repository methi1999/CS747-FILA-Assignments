import argparse
import numpy as np

"""
Actions: 0-left, 1-top, 2-right, 3-bottom
Rewards: 10^5 for reaching end, else 0
"""

def reward(grid_val):
    if grid_val == 0 or grid_val == 2:
        return 0
    elif grid_val == 3:
        return 1e5
    else:
        raise Exception("Hit a wall")


def grid_to_mdp(filepath):
    with open(filepath, 'r') as f:
        data = [x.strip() for x in f.readlines()]

    grid = []
    for _, data in enumerate(data):
        cur = []
        for _, val in enumerate(data.split(' ')):
            cur.append(int(val))
        grid.append(cur)
    
    grid = np.array(grid)
    num_states = 0
    transitions = []
    state_matrix = -np.ones(grid.shape, dtype=np.int64)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 1:
                state_matrix[i, j] = num_states
                num_states += 1

    start_state = state_matrix[np.where(grid == 2)][0]
    end_state = state_matrix[np.where(grid == 3)][0]
    wall_reward = 0
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cur_state = state_matrix[i, j]
            t = []
            if grid[i, j] == 0 or grid[i, j] == 2: # empty or start tile
                # south
                if i+1 < grid.shape[0] and grid[i+1, j] != 1:
                    t.append((cur_state, 2, state_matrix[i+1,j], reward(grid[i+1, j])))
                else:
                    t.append((cur_state, 2, cur_state, wall_reward))
                # east
                if j+1 < grid.shape[1] and grid[i, j+1] != 1:
                    t.append((cur_state, 1, state_matrix[i,j+1], reward(grid[i, j+1])))
                else:
                    t.append((cur_state, 1, cur_state, wall_reward))
                # west
                if j-1 >= 0 and grid[i, j-1] != 1:
                    t.append((cur_state, 3, state_matrix[i,j-1], reward(grid[i, j-1])))
                else:
                    t.append((cur_state, 3, cur_state, wall_reward))
                # north
                if i-1 >= 0 and grid[i-1, j] != 1:
                    t.append((cur_state, 0, state_matrix[i-1,j], reward(grid[i-1, j])))
                else:
                    t.append((cur_state, 0, cur_state, wall_reward))
                                
            # elif grid[i, j] == 3: # end tile
            #     t.append((cur_state, 0, cur_state, 0))
            #     t.append((cur_state, 1, cur_state, 0))
            #     t.append((cur_state, 2, cur_state, 0))
            #     t.append((cur_state, 3, cur_state, 0))
            
            transitions += t
    
    final = ''
    final += 'numStates {}\n'.format(num_states)
    final += 'numActions {}\n'.format(4)
    final += 'start {}\n'.format(start_state)
    final += 'end {}\n'.format(end_state)
    for s1, a, s2, r in transitions:
        final += 'transition {} {} {} {} 1\n'.format(s1, a, s2, r)
    final += 'mdptype episodic\n'
    final += 'discount 0.9'
    print(final)


if __name__ == "__main__":
    # argparse object
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid",type=str)
    args = parser.parse_args()

    grid_to_mdp(args.grid)