from grid_world import GridWorld
import numpy as np

class Mario:
    def __init__(self, env, gamma=0.9):
        """
        Arguments:
            env: object of class "GridWorld" over which agent "Mario" is intended to learn policy on
            gamma: value discount you want the agent "Mario" to learn the state-values with

        Default Policy:
            B: blocked states (default env setup)
            v: move down

            : v | v | v | END
            : v | B | v | END
            : v | v | v | v   
        """
        self.gamma = gamma
        self.env = env
        self.state_values = {
            state: 0.0
            for state in self.env.valid_states
        }
        self.policy = {
            state: (((0,-1),'down') if state not in self.env.end_states else ((0,0),'END'))
            for state in self.env.valid_states
        } # necessary to have default policy as the first action in action space for correct visualization

    def show_state_values(self):
        for j in range(self.env.ydim, 0, -1):
            for i in range(1, self.env.xdim+1):
                if (i,j) in self.state_values:
                    print(f'{self.state_values[(i,j)]:+.6f}', end='|')
                else:
                    print("         ", end="|")
            print()

    def show_policy(self):
        for j in range(self.env.ydim, 0, -1):
            for i in range(1, self.env.xdim+1):
                if (i,j) in self.policy:
                    _, action_name = self.policy[(i,j)]
                    print(f'{action_name:5s}', end='|')
                else:
                    print("     ", end="|")
            print()