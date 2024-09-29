import numpy as np

class GridWorld:
    def __init__(self, x=4, y=3, blocked_states=[(2,2)], end_states=[(4,3),(4,2)], controller_reliability=0.8):
        """
        Arguments:
            x: length of gridworld
            y: width of gridworld
            blocked_states: list of states blocked from access to agent "Mario"
            end_states: list of states where the game terminates
            controller_reliability: reliablity of controller expressed as probability

        Default rewards:
            B: blocked states (default env setup)

            : 0 | 0 | 0 | +1
            : 0 | B | 0 | -1 
            : 0 | 0 | 0 | 0    
        """
        self.xdim = x
        self.ydim = y
        self.blocked_states = blocked_states # list of states (as tuples) that cannot be accessed by agent
        self.end_states = end_states # list of end states (as tuples) --> for not evaluating state values here
        self.prob = controller_reliability
        # valid states
        self.valid_states = [
            (i+1, j+1) 
            for j in range(self.ydim) 
            for i in range(self.xdim) 
            if (i+1, j+1) not in self.blocked_states
        ]
        # Action space
        self.action_space = {(0,-1): 'down', (-1,0): 'left', (1,0): 'right', (0,1): 'up'}
        # Rewards
        self.rewards = {
            state: 0.0
            for state in self.valid_states
        }
        reward_dict = {(4,2): -1, (4,3): 1}
        self.set_rewards(reward_dict=reward_dict, transition_reward=0.0)

    # def reshape_index(self, coords):
    #     x, y = coords
    #     return self.ydim-y, x-1
    
    def set_rewards(self, reward_dict, other_states = 0.0, transition_reward = 0.0):
        """
        Optional setter:
            def set_rewards(reward_dict, other_states = 0.0, transition_reward = 0.0)
                reward_dict: {key = state: value = intended reward for state}
                other_states: intended reward for states other than those mentioned in reward_dict
                transition_reward: reward for transition from current_state(s) to new_state(s') --> common for all states

        """
        for i, state in enumerate(self.valid_states):
            if state in reward_dict:
                self.rewards[state] = reward_dict[state] + transition_reward
            else:
                self.rewards[state] = other_states + transition_reward

    def show_rewards(self):
        for j in range(self.ydim, 0, -1):
            for i in range(1, self.xdim+1):
                if (i,j) in self.rewards:
                    print(f'{self.rewards[(i,j)]:+.2f}', end='|')
                else:
                    print("     ", end="|")
            print()

    def transition_probs(self, state, action):
        """Take any (state, action) pair and return transition probabilities to all valid transition states"""
        # transverse actions (in list form) 
        transverse_action_1 = np.flip(action)
        transverse_action_2 = [-i for i in transverse_action_1]
        # New states
        action_state = tuple(map(sum, zip(state, action))) # new state when action executed as intended
        flip_1_state = tuple(map(sum, zip(state, tuple(transverse_action_1))))
        flip_2_state = tuple(map(sum, zip(state, tuple(transverse_action_2))))
        # check and correct if new states are out of bounds
        action_state = action_state if action_state in self.valid_states else state
        flip_1_state = flip_1_state if flip_1_state in self.valid_states else state
        flip_2_state = flip_2_state if flip_2_state in self.valid_states else state
        return {action_state: self.prob, flip_1_state: 0.5*(1-self.prob), flip_2_state: 0.5*(1-self.prob)}