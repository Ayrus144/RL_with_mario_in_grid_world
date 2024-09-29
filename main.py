from grid_world import GridWorld
from mario import Mario
from iter_schemes import Iter
from animator import Animation

# Default setup of the problem statement seems to be 'ill-posed'
# (Can give different results due to machine precision)

# We encourage to play with the initialization of the objects below 
# by exploring the corresponding source code files mentioned beside them.
env   = GridWorld()                 # Check gridworld.py 
agent = Mario(env=env)              # Check mario.py 
learn = Iter(env=env, agent=agent)  # Check iter_schemes.py 

print('*** Env rewards ***')
env.show_rewards()

# Only change the 'value_iter' variable according to your preference for iteration scheme
# which will also set 'poiicy_iter' variable for the opposite behaviour
value_iter = False
policy_iter = not value_iter
animate = True

if value_iter:
    print("*** Starting value iter ***")
    gen = learn.by_value_iter(anim=animate)
    if animate:
        animator = Animation(env, agent, gen, 'value_iter')
        animator.animate()
    print("*** Value iter convergence reached ***")

if policy_iter:
    print('*** Initial policy ***')
    agent.show_policy()
    print("*** Starting policy iter ***")
    gen = learn.by_policy_iter(anim=animate)
    if animate:
        animator = Animation(env, agent, gen, 'policy_iter')
        animator.animate()
    print("*** Policy iter convergence reached ***")

print('--> Optimal State values')
agent.show_state_values()
print('--> Optimal Policy')
agent.show_policy()