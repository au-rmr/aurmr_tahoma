import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import rospkg
from copy import copy
    

def plot_score(x:np.array, y:np.array, z:np.array, x_step, y_step):
    ''' Generate 3D barplot
        args:
        x, y - coordinates of the pod (in meters)
        z - success score in range[0, 1] '''
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    color = np.zeros(shape=(z.shape[0], 4), dtype=float)
    color[:, 3] = 1 # All RGBA Alpha-value = 1
    for i, score in enumerate(z):
        color[i, 0] = score # Higher scores result in more red bars
        color[i, 2] = 1 - score # Lower scores result in more blue bars
    for i in range(z.shape[0]):
        bar = ax.bar3d(x=x[i], y=y[i], z=np.zeros_like(z), dx=x_step/2, dy=y_step/2, dz=z[i]) # Create the 3D bars
        bar.set(color=color[i], edgecolor='black', linewidth=0.5)

    ax.view_init(elev=45, azim=135) # Set view point in degrees
    # Set labels for x, y, and z axes
    ax.set_title(label="Reachability score as a function of pod's positions", fontsize='small')
    ax.set_xlabel(xlabel="Pod's X-coordinate", fontsize='x-small')
    ax.set_ylabel(ylabel="Pod's Y-coordinate", fontsize='x-small')
    ax.set_zlabel(zlabel="Success score", fontsize='x-small')
    # Change tick properties (labelsize, interval)
    ax.tick_params(axis='x', labelsize=4, rotation=45.0)
    ax.tick_params(axis='y', labelsize=4)
    ax.tick_params(axis='z', labelsize=4)
    ax.set_xticks(ticks=np.arange(x.min(), x.max() + 0.001, x_step))
    ax.set_yticks(ticks=np.arange(y.min(), y.max() + 0.001, y_step))
    ax.set_zticks(ticks=np.arange(0.0, 1.0, 0.1))
    
    plt.show()
    

if __name__ == '__main__':
    x_step = 0.025
    y_step = 0.080
    x = []
    y = []
    scores = []
    rospack = rospkg.RosPack()
    filename = rospack.get_path('aurmr_tasks') + '/scripts/successScores_600-1000-25_220-940-80'
    with open(filename + '.txt', 'r') as file:
        for line in file:
            if len(line.rstrip()) != 0: # Skip '\n' lines
                x.append(float(line[1 : line.find(',')]))
                y.append(float(line[line.find(',') + 1 : line.find(']')]))
                scores.append(float(line[line.find(':') + 1 : line.find('n')])/100)

    plot_score(x=np.array(x), y=np.array(y), z=np.array(scores), 
               x_step=x_step, y_step=y_step)
