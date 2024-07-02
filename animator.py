import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.animation import FuncAnimation

class Animation:
    def __init__(self, env, agent, gen, filename):
        self.env = env
        self.agent = agent
        self.root_gen = gen
        self.filename = filename

        # Create the figure and axis
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10,6), layout='constrained')
        self.fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.3, wspace=0, hspace=0)
        self.set_axis(self.ax1, 'State Values')
        self.set_axis(self.ax2, 'Policy')

        # Create the grid of text elements
        self.grid_state_values = np.empty((self.env.ydim, self.env.xdim), dtype=object)
        for j in range(self.env.ydim, 0, -1):
            for i in range(1, self.env.xdim+1):
                if (i,j) in self.agent.state_values:
                    self.grid_state_values[self.env.ydim-j, i-1] = self.ax1.text(
                        i-1,j-1, 
                        "{:.3f}".format(self.agent.state_values[(i,j)]), 
                        ha='center', va='center', fontsize=20
                        )
                else:
                    self.grid_state_values[self.env.ydim-j, i-1] = self.ax1.text(
                        i-1,j-1, 
                        '', 
                        ha='center', va='center', fontsize=20
                        )
        self.grid_policy = np.empty((self.env.ydim, self.env.xdim), dtype=object)
        for j in range(self.env.ydim, 0, -1):
            for i in range(1, self.env.xdim+1):
                if (i,j) in self.agent.policy:
                    _, action_name = self.agent.policy[(i,j)]
                    self.grid_policy[self.env.ydim-j, i-1] = self.ax2.text(
                        i-1,j-1, 
                        action_name, 
                        ha='center', va='center', fontsize=20
                        )
                else:
                    self.grid_policy[self.env.ydim-j, i-1] = self.ax2.text(
                        i-1,j-1, 
                        '', 
                        ha='center', va='center', fontsize=20
                        )

    def animate(self):
        # Set up the animation
        self.ani = FuncAnimation(
            self.fig, self.update_value_policy, 
            frames=self.root_gen,
            init_func=self.init_value_policy, 
            save_count=10000,
            interval=2000, repeat=False
            )
        # self.ax1.grid(True, linewidth = 2)
        # self.ax2.grid(True, linewidth = 2)
        self.ani.save('{}.gif'.format(self.filename), writer='Pillow', fps=0.5,)
        # Show the animation
        # plt.show()

    def init_value_policy(self):
        self.fig.suptitle(f'Initialization', fontsize =30)
        # ax1 - state values
        for j in range(self.env.ydim, 0, -1):
            for i in range(1, self.env.xdim+1):
                if (i,j) in self.agent.state_values:
                    self.grid_state_values[self.env.ydim-j, i-1].set_text("{:.3f}".format(0))
                else:
                    self.grid_state_values[self.env.ydim-j, i-1].set_text('')
        
        # ax2 - policy
        for j in range(self.env.ydim, 0, -1):
            for i in range(1, self.env.xdim+1):
                if (i,j) in self.agent.policy:
                    _, action_name = self.agent.policy[(i,j)]
                    self.grid_policy[self.env.ydim-j, i-1].set_text(action_name)
                else:
                    self.grid_policy[self.env.ydim-j, i-1].set_text('')
        self.color_policy()


    def update_value_policy(self, frame):
        self.fig.suptitle(f'{frame}', fontsize =30)
        # ax1 - state value
        for j in range(self.env.ydim, 0, -1):
            for i in range(1, self.env.xdim+1):
                if (i,j) in self.agent.state_values:
                    self.grid_state_values[self.env.ydim-j, i-1].set_text("{:.3f}".format(self.agent.state_values[(i,j)]))
                else:
                    self.grid_state_values[self.env.ydim-j, i-1].set_text('')

        # ax2 - policy
        for j in range(self.env.ydim, 0, -1):
            for i in range(1, self.env.xdim+1):
                if (i,j) in self.agent.policy:
                    _, action_name = self.agent.policy[(i,j)]
                    self.grid_policy[self.env.ydim-j, i-1].set_text(action_name)
                else:
                    self.grid_policy[self.env.ydim-j, i-1].set_text('')
        self.color_policy()

    def color_policy(self):
        values, policy, arrows = self.get_data()

        # New colormap with vmin set to white
        cmap = cm.coolwarm(range(256))
        cmap[0,] = np.array([0., 0., 0., 0.])

        cmap = colors.ListedColormap(cmap)
        norm = colors.Normalize(vmin =-1, vmax =1)
        self.ax1.imshow(values, cmap, norm)

        cmap = colors.ListedColormap(['white','grey','dodgerblue'])
        norm = colors.BoundaryNorm([-1,0,0.5,1], cmap.N)
        self.ax2.imshow(policy, cmap, norm)
        x, y = zip(*arrows)
        x = np.array(x)
        y = np.array(y)
        self.ax2.quiver(
            x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], 
            scale_units='xy', angles='xy', scale=2,
            color='indianred', width=0.1, headaxislength=5, headwidth=5
            )

    def get_data(self):
        xdim = self.env.xdim
        ydim = self.env.ydim
        policy = np.zeros((ydim, xdim))
        values = np.empty((ydim, xdim))
        arrows = []

        for j in range(ydim, 0, -1):
            for i in range(1, xdim+1):
                if (i,j) in self.env.valid_states:
                    values[j-1, i-1] = self.agent.state_values[(i,j)]
                else:
                    values[j-1, i-1] = -1
        for (i,j) in self.env.blocked_states:
            policy[j-1, i-1] = -1

        goto = (1,1)
        action_name = 'down'
        iter = 0
        while action_name != 'END': # and iter < xdim*ydim:
            iter += 1
            x,y = goto
            arrows.append((x-1,y-1))
            policy[y-1, x-1] = 1
            action, action_name = self.agent.policy[goto]
            new_state = tuple(map(sum, zip(goto, action)))
            if new_state in self.env.valid_states:
                goto = new_state
            else:
                # arrows.append((x-1,y-1))
                break
        return values, policy, arrows

    def set_axis(self, ax, title):
        ax.set_xlim(-0.5, self.env.xdim - 0.5)
        ax.set_ylim(-0.5, self.env.ydim - 0.5)
        ax.set_xticks(np.arange(0.5, self.env.xdim - 0.5, 1))
        ax.set_yticks(np.arange(0.5, self.env.ydim - 0.5, 1))
        ax.set_title(f'{title}', fontsize=25)
        ax.grid(True, linewidth = 2, color='black')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        for loc in ["top", "bottom", "left", "right"]:
            ax.spines[loc].set_linewidth(2)