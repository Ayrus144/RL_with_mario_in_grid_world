from grid_world import GridWorld
from mario import Mario
from iter_schemes import Iter
from animator import Animation

env   = GridWorld()
agent = Mario(env=env) 
learn = Iter(env=env, agent=agent)
# Default setup of the problem statement seems to be ill-posed

print('*** Env rewards ***')
env.show_rewards()

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