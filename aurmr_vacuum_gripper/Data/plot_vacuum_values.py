
## Code to plot vacuum measurements together

import numpy as np
import matplotlib.pyplot as plt

file_names = [
                # '12-5-24_unbostructred_2',
                # '12-5-24_flat_object_2',
                # '12-5-24_object_distant',
                # '12-5-24_object_near',
                # '12-5-24_suction_light_flat_object',
                # '12-5-24_suction_heavy_flat_object',
                # '12-5-24_suction_heavy_flat_object_suction_cup_uncompressed',
                # '12-5-24_suction_heavy_flat_object_suction_cup_compressed'
                '12-5-24_suction_unobstructed',
                '12-5-24_suction_on_flat_object',
                '12-5-24_suction_on_cyl',
                '12-5-24_cyl_gap',
                '12-5-24_bag'
                ]
plot_names = [
                # 'Unobstructed_2',
                # 'Flat_object_2',
                # 'Suction light flat object',
                # 'Suction heavy flat object',
                # 'Suction heavy flat object suction cup uncompressed',
                # 'Suction heavy flat object suction cup compressed'
                'Unobstructed',
                'Attached to flat object',
                'Attached to cylindrical object',
                'Attached to cylindrical object with lid gap',
                'Attached to bag'
                ]

# Create a figure and axis
plt.figure()

# Loop through data files and plot
for i in range(5):
    data = np.load(f'{file_names[i]}.npy')
    plt.plot(data[0], data[1], label=plot_names[i])

# Add labels and title
plt.xlabel('Time, sec')
plt.ylabel('Vacuum, mbar')
# plt.title('Vacuum Multiple Scenarios')
plt.title('Vacuum Multiple Scenarios')

# Add legend after all plots
plt.legend()

# Save the figure
# plt.savefig('multiple_scenarios.png')
plt.savefig('multiple_scenarios_2.png')


# Show the plot (optional)
plt.show()

## Code to plot difference below

# import numpy as np
# import matplotlib.pyplot as plt

# file_names = [
#     '12-5-24_suction_unobstructed',
#     '12-5-24_suction_on_flat_object',
#     '12-5-24_suction_on_cyl',
#     '12-5-24_cyl_gap',
#     '12-5-24_bag'
# ]

# # Load data
# flat_object_data = np.load(f'{file_names[1]}.npy')  # Flat object
# cylinder_data = np.load(f'{file_names[2]}.npy')     # Cylindrical object
# bag_data = np.load(f'{file_names[4]}.npy')          # Bag

# # Ensure the time arrays are aligned
# time = flat_object_data[0]  # Assuming the time axis is the same for all
# flat_vacuum = flat_object_data[1]
# cylinder_vacuum = cylinder_data[1]
# bag_vacuum = bag_data[1]

# # Calculate differences
# diff_flat_cyl = flat_vacuum - cylinder_vacuum
# diff_flat_bag = flat_vacuum - bag_vacuum

# # Plot the differences
# plt.figure()

# # Plot difference between flat object and cylindrical object
# plt.plot(time, diff_flat_cyl, label='Flat object - Cylindrical object')

# # Plot difference between flat object and bag
# plt.plot(time, diff_flat_bag, label='Flat object - Bag')

# # Add labels, title, and legend
# plt.xlabel('Time, sec')
# plt.ylabel('Vacuum difference, mbar')
# plt.title('Vacuum Difference: Flat Object vs Others')
# plt.legend()

# # Save and show the plot
# plt.savefig('vacuum_differences.png')
# plt.show()
