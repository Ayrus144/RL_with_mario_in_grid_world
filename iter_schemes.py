from grid_world import GridWorld
from mario import Mario
import copy
import numpy as np

class Iter:
    def __init__(self, env, agent, tol = 0.000001):
        """
        Arguments:
            env: object of class "GridWorld" over which agent "Mario" is intended to learn policy on
            agent: object of class "Mario" who learns the policy
            tol: tolerance under which the difference between state-values during value iterations is considered converged
            NOTE: value iteration step is done during both value_iteration method and policy iteration method
        """
        self.env = env
        self.agent = agent
        self.tolerance = tol

    def expected_Q_value(self, state, action):
        """
        Helper function: Used for helper functions "state_value_greedy" and "state_value_policy"
        Take any (state, action) pair and return expected Q value based on (expected) state values in previous iteration
        """
        q_value = 0
        if state in self.env.end_states:
            return q_value
        
        transition_probs = self.env.transition_probs(state, action)
        for new_state in transition_probs:
            prob = transition_probs[new_state]
            reward = self.env.rewards[new_state]
            gamma = self.agent.gamma
            # state value at the new_state (newly possible transition state) as calulated in previous iteration
            prev_iter_state_value = self.agent.state_values[new_state] 
            q_value += prob * (reward + gamma* prev_iter_state_value)
        # Gauss seidel variety would be wrong (unstable)
        # q_val += prob*(self.env.rewards[new_state] + gamma*self.agent.state_value[new_state])
        return q_value

    def state_value_greedy(self, state):
        """
        Helper function: Used for helper function "state_value_policy" and function "by_value_iter"
        Take any state and return the maximum Q-value and corresponding action as a pair
        """
        state_value = 0
        best_action = 'bug'
        Q_values= {}
        for action in self.env.action_space:
            q_value = self.expected_Q_value(state, action)
            Q_values[action] = q_value
        best_action, state_value = max(Q_values.items(), key=lambda k: k[1]) # max key-value pair
        return state_value, best_action # return best action too
    
    def state_value_policy(self, state):
        """
        Helper function: Used for function "by_policy_iter"
        Take any state and return the 
            Q-value based on (state, action) pair using current policy, and 
            best/ greedy action (corresponding to maximum Q-value) 
        as a pair
        """
        action, _ = self.agent.policy[state]
        policy_value = self.expected_Q_value(state, action)
        _, best_action = self.state_value_greedy(state)
        return policy_value, best_action # return best action too
    
    def by_value_iter(self, show_updates = False, anim = False):
        """
        Calculate Optimal Policy using Value Iteration
        """
        iter = 0
        val_error = np.ones(len(self.env.valid_states))
        while val_error.max() > self.tolerance: 
            curr_iter_state_values = {}
            updated_policy = {}
            for i, state in enumerate(self.env.valid_states):
                if state in self.env.end_states:
                    curr_iter_state_values[state] = 0
                    updated_policy[state] = ((0,0),'END')
                    val_error[i] = 0
                else:
                    # get best Q-value action pair
                    state_value, action = self.state_value_greedy(state)
                    # store in the answers in temp dicts
                    curr_iter_state_values[state] = state_value
                    updated_policy[state] = (action, self.env.action_space[action])
                    # update convergence list
                    val_error[i] = abs(state_value - self.agent.state_values[state])
            self.agent.state_values = curr_iter_state_values
            self.agent.policy = updated_policy
            # print log
            iter += 1
            print(f'--> Iteration {iter}')
            if show_updates:
                self.agent.show_state_values()
                self.agent.show_policy()
            # wait for animation
            if anim:
                yield "Iter: {}".format(iter)
            # Policy extraction, in a way, is done once for every loop over the states
            # We just save the current policy (as the while loop can end any time)
            # We don't use this policy in our subsequent calculation anywhere

    def by_policy_iter(self, show_updates = False, anim=False):
        """
        Calculate Optimal Policy using Policy Iteration
        """
        epoch = 0 # counts number of policies tried
        total_steps = 0
        while True:
            epoch += 1
            iter = 0
            updated_policy = {}
            val_error = np.ones(len(self.env.valid_states))
            while val_error.max() > self.tolerance:
                curr_iter_state_values = {}
                for i, state in enumerate(self.env.valid_states):
                    if state in self.env.end_states:
                        curr_iter_state_values[state] = 0
                        updated_policy[state] = ((0,0),'END')
                        val_error[i] = 0
                    else:
                        # get predicted Q-value and best action
                        policy_value, action = self.state_value_policy(state)
                        # store in the answers in temp dicts
                        curr_iter_state_values[state] = policy_value
                        updated_policy[state] = (action, self.env.action_space[action])
                        # update convergence list
                        val_error[i] = abs(policy_value - self.agent.state_values[state])
                self.agent.state_values = curr_iter_state_values
                # print log
                iter += 1
                total_steps += 1
                print(f'Epoch {epoch} --> Iteration {iter} | Total steps {total_steps}')
                if show_updates:
                    self.agent.show_state_values()
                    self.agent.show_policy()
                # wait for animation
                if anim:
                    yield "Epoch: {}, Iter: {}, Steps: {}".format(epoch, iter, total_steps)
                # Policy improvement, in a way, is done once for every loop over the states
                # We just save the current best policy (as the while loop can end any time)
                # We don't use this policy in our subsequent calculation untill next loop
            if updated_policy != self.agent.policy:
                self.agent.policy = copy.deepcopy(updated_policy)
                print(f' --> Updated policy for Epoch {epoch+1}')
                self.agent.show_policy()
            else:
                break
    
# IGNORE: used for testing
if __name__ == "__main__":
    # Hover over the class initializers for more info (VS code)
    env   = GridWorld()
    agent = Mario(env=env) 
    learn = Iter(env=env, agent=agent)
    # Default setup of the problem statement seems to be ill-posed

    print('*** Env rewards ***')
    env.show_rewards()

    value_iter = False
    policy_iter = not value_iter

    if value_iter:
        print("*** Starting value iter ***")
        learn.by_value_iter()
        print("*** Value iter convergence reached ***")
    
    if policy_iter:
        print('*** Initial policy ***')
        agent.show_policy()
        print("*** Starting policy iter ***")
        learn.by_policy_iter()
        print("*** Policy iter convergence reached ***")

    print('--> Optimal State values')
    agent.show_state_values()
    print('--> Optimal Policy')
    agent.show_policy()