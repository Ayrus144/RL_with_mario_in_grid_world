import numpy as np
import copy

class GridWorld:
    def __init__(self, x=4, y=3):
        self.xdim = x
        self.ydim = y
        self.blocked_states = [(2,2)] # states that cannot be accessed by agent
        self.end_states = [(4,3),(4,2)] # end states --> for not evaluating state values here
        # valid states
        self.valid_states = [
            (i+1, j+1) 
            for j in range(self.ydim) 
            for i in range(self.xdim) 
            if (i+1, j+1) not in self.blocked_states
        ]
        self.policy = {valid_state: 'bug' for valid_state in self.valid_states} 

        # Edge states
        # self.left_edge = [(0,j+1) for j in range(self.ydim)]
        # self.right_edge = [(self.xdim+1,j+1) for j in range(self.ydim)]
        # self.bottom_edge = [(i+1, 0) for i in range(self.xdim)]
        # self.top_edge = [(i+1, self.ydim+1) for i in range(self.xdim)]
        # self.edge_states = self.left_edge + self.right_edge + self.bottom_edge + self.top_edge 

        # Actions
        self.move_left = (-1,0)
        self.move_right = (1,0)
        self.move_up = (0,1)
        self.move_down = (0,-1)
        self.actions = [self.move_left, self.move_right, self.move_up, self.move_down]
        # Rewards and State Values
        self.rewards = np.zeros((self.ydim, self.xdim))
        self.state_values = np.zeros((self.ydim, self.xdim))
    
    # Environement
    def set_blocked_states(self, state_list):
        self.blocked_states = state_list

    def show_valid_states(self):
        print("\n*** valid states ***")
        print(*self.valid_states)

    def show_edge_states(self):
        print("\n*** edge states ***")
        print(*self.edge_states)

    def reshape_index(self, coords):
        x, y = coords
        return self.ydim-y, x-1

    # Rewards
    def show_rewards(self):
        print("\n*** grid rewards ***")
        for i in range(self.ydim):
            row = [str(rew) for rew in self.rewards[i]]
            print("|".join(row))

    def get_reward(self, coords):
        x,y = self.reshape_index(coords)
        return self.rewards[x, y]
    
    def set_reward(self, coords, val):
        x,y = self.reshape_index(coords)
        self.rewards[x,y] = val
        print(f'reward at {coords} set to {self.get_reward(coords)}')

    # State values
    def show_state_values(self, des=True):
        if des:
            print("*** GridWorld state values ***")
        for i in range(self.ydim):
            row = ['{0:.6f}'.format(rew) for rew in self.state_values[i]]
            print("|".join(row))

    def get_state_value(self, coords):
        x,y = self.reshape_index(coords)
        return self.state_values[x, y]  
    
    def set_state_value(self, coords, val):
        x,y = self.reshape_index(coords)
        self.state_values[x,y] = val

    # Policy
    def show_policy(self):
        for j in range(self.ydim, 0, -1):
            for i in range(1, self.xdim+1):
                if (i,j) in self.policy:
                    print(f'{self.policy[(i,j)]:5s}', end='|')
                else:
                    print("     ", end="|")
            print()


# Create and set environemnt
env = GridWorld()
block_states = [(2,2)]
env.set_blocked_states(block_states)
env.show_valid_states()
# env.show_edge_states()

# Set Rewards R(s,a,s'): 
#   Given when agent reaches the corresponding state 
#   from any previous state (irrespective of action)
env.set_reward((4,3), 1)
env.set_reward((4,2), -1)
env.show_rewards()

# VALUE ITERATION
def calculate_Q_value(state, action, curr_state_values, gamma=0.9):
    # Skip end-states value calculation
    q_val = 0
    if state in env.end_states:
        return q_val
    # transverse actions (in list form) 
    transverse_action_1 = np.flip(action)
    transverse_action_2 = [-i for i in transverse_action_1]
    # New states
    action_state = tuple(map(sum, zip(state, action))) # new state when action executed as intended
    flip_1_state = tuple(map(sum, zip(state, tuple(transverse_action_1))))
    flip_2_state = tuple(map(sum, zip(state, tuple(transverse_action_2))))
    action_prob = {action_state: 0.8, flip_1_state: 0.1, flip_2_state: 0.1}
    # Calculation begin
    for new_state in action_prob:
        prob = action_prob[new_state]
        # check if valid state
        if new_state not in env.valid_states:
            new_state = state
        new_state_value = curr_state_values[env.reshape_index(new_state)]
        q_val += prob*(env.get_reward(new_state) + gamma*new_state_value)
        # Gauss seidel variety --> would be wrong
        # q_val += prob*(env.get_reward(new_state) + gamma*env.get_state_value(new_state))
    
    return q_val

# Create and set params/ emthods for iteration
tol = 0.000001
val_error = np.ones(len(env.valid_states))
itr = 0

# Start Value Improvement
print("\n*** Starting Value Iteration ***")
while val_error.max() > tol:
    itr += 1
    curr_state_values = copy.deepcopy(env.state_values)
    for i,state in enumerate(env.valid_states):
        curr_value = env.get_state_value(state)
        # calc new state value
        Q_values = {}
        for action in env.actions:
            q_val = calculate_Q_value(state, action, curr_state_values)
            Q_values[action] = q_val
        max_val_action, new_value = max(Q_values.items(), key=lambda k: k[1]) # max key-value pair

        # Policy extraction
        # Keep saving Policy every iteration, since we don't know when the iteration will converge
        if state in env.end_states:
            env.policy[state] = 'none'
        else:
            if max_val_action == env.move_left:
                env.policy[state] = 'left'
            elif max_val_action == env.move_right:
                env.policy[state] = 'right'
            elif max_val_action == env.move_up:
                env.policy[state] = 'up'
            elif max_val_action == env.move_down:
                env.policy[state] = 'down'
            else:
                env.policy[state] = 'bug'

        val_error[i] = abs(new_value - curr_value)
        env.set_state_value(state, new_value)

    print(f'--> Iteration {itr}')
    env.show_state_values(des = False)
    # print(env.policy)

# Finally, what we need from Value Iteration
print('\n*** Optimal Value ***')
env.show_state_values(des = False)
print('\n*** Optimal policy ***')
print(env.policy)
env.show_policy()

# ROUGH WORK

# curr_state_values = copy.deepcopy(env.state_values)
# x,y = env.reshape_index((4,2))
# curr_state_values[x,y] = 10
# print(curr_state_values)
# print(env.state_values)

# a = (1,2)
# env.state_values[1,1] = 10
# # x,y = env.reshape_index((2,2))
# # print(x,y)
# print(env.state_values[env.reshape_index((2,2))])