import argparse
import numpy as np

if __name__ == "__main__":
    # argparse object
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid",type=str)    
    parser.add_argument("--value_policy",type=str)
    args = parser.parse_args()

    with open(args.grid, 'r') as f:
        grid_data = [x.strip() for x in f.readlines()]    
    
    grid = []
    for row, data in enumerate(grid_data):
        cur = []
        for col, val in enumerate(data.split(' ')):
            cur.append(int(val))
        grid.append(cur)
    
    grid = np.array(grid)

    num_states = 0
    state_matrix = -np.ones(grid.shape, dtype=np.int64)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 1:
                state_matrix[i, j] = num_states
                num_states += 1

    start_x, start_y = np.where(grid == 2)
    end_x, end_y = np.where(grid == 3)
    # print(start_x, start_y, end_x, end_y)

    with open(args.value_policy, 'r') as f:
        policy = [x.strip() for x in f.readlines()]
    
    moves = ''
    cur_x, cur_y = start_x[0], start_y[0]
    end_x, end_y = end_x[0], end_y[0]
    
    while cur_x != end_x or cur_y != end_y:
        cur_state_id = state_matrix[cur_x, cur_y]        
        move_id = int(policy[cur_state_id].split(' ')[-1])
        # print(cur_x, cur_y, move_id, cur_state_id)
        if move_id == 0:
            cur_x -= 1
            move = 'N'
        elif move_id == 1:
            cur_y += 1
            move = 'E'
        elif move_id == 2:
            cur_x += 1
            move = 'S'
        else:
            cur_y -= 1
            move = 'W'
        
        moves += move + ' '
    
    print(moves[:-1])