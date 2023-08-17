import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import rospkg
from copy import copy
import os
    

def plot_score(x:np.array, y:np.array, z:np.array, x_step, y_step):
    ''' Generate 3D barplot
        args:
            x, y - coordinates of the pod (in meters)
            z - success score in range[0.0000, 1.0000] '''
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Normalize data for colormap
    norm = plt.Normalize(min(z), max(z))
    colors = plt.cm.rainbow(norm(z)) # viridis, rainbow, brg

    bar = ax.bar3d(x, y, np.zeros_like(z), x_step/2, y_step/2, z, shade=True, color=colors) # Create the 3D bars
    # bar.set(color=color[i], edgecolor='black', linewidth=0.5)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.rainbow)
    mappable.set_array(z)
    cbar = plt.colorbar(mappable, ax=ax, location='left', aspect=50, format='%7.4f')
    cbar.set_label(label='Success score', fontsize='xx-small')
    cbar.set_ticks(np.linspace(min(z), max(z), 10, endpoint=True))
    cbar.ax.tick_params(labelsize=5)

    ax.view_init(elev=45, azim=135) # Set view point in degrees

    # Set labels for x, y, and z axes
    ax.set_title(label="Reachability score as a function of pod's positions", fontsize='x-small')
    ax.set_xlabel(xlabel="Pod's X-coordinate", fontsize='xx-small')
    ax.set_ylabel(ylabel="Pod's Y-coordinate", fontsize='xx-small')
    # ax.set_zlabel(zlabel="Success score", fontsize='xx-small')

    # Change tick properties (labelsize, interval)
    ax.tick_params(axis='x', labelsize=3.5, rotation=45.0)
    ax.tick_params(axis='y', labelsize=3.5, rotation=-45.0)
    # ax.tick_params(axis='z', labelsize=0)
    ax.set_xticks(ticks=np.arange(x.min(), x.max() + 0.001, x_step))
    ax.set_yticks(ticks=np.arange(y.min(), y.max() + 0.001, y_step))
    ax.set_zticks(ticks=[])

    ax.text(x=0.52, y=0.2, z=2.2, s='Current [x, y, score]:\n[0.832, 0.423, 0.9844]', fontsize='xx-small', va='top', ha='right')
    ax.text(x=0.515, y=0.195, z=2, s='Highest [x, y, score]:\n[0.750, 0.420, 0.9994]\n[0.950, 0.470, 0.9981]\n[0.750, 0.540, 0.9975]\n[0.840, 0.470, 0.9975]\n[0.920, 0.400, 0.9975]', 
            fontsize='xx-small', va='top', ha='right')

    # ax.text(x=0.52, y=0.2, z=2.2, s='Highest [x, y, score]:\n[0.850, 0.380, 0.9978]\n[0.910, 0.420, 0.9978]\n[0.920, 0.370, 0.9978]\n[0.970, 0.390, 0.9978]\n[0.910, 0.410, 0.9963]\n[0.930, 0.370, 0.9963]', 
    #         fontsize='xx-small', va='top', ha='right')
    
    
    plt.show()
    

if __name__ == '__main__':
    x_step = 0.010
    y_step = 0.010
    x = []
    y = []
    scores = []
    rospack = rospkg.RosPack()
    filename = './successScores1_720-960-10_360-580-10' # './successScores23_720-970-10_360-580-10'
    with open(filename + '.txt', 'r') as file:
        for line in file:
            if len(line.rstrip()) != 0: # Skip '\n' lines
                x.append(float(line[1 : line.find(',')]))
                y.append(float(line[line.find(',') + 1 : line.find(']')]))
                scores.append(float(line[line.find(':') + 1 : line.find('n')])/100)

    plot_score(x=np.array(x), y=np.array(y), z=np.array(scores), 
               x_step=x_step, y_step=y_step)
