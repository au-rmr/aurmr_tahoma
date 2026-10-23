import matplotlib.pyplot as plt
import os 
import time

while True:
    file_path = '/home/aurmr/workspaces/thomas_ws/tmp/cur_mask.png'
    while not os.path.exists(file_path):
        time.sleep(1)
    img = plt.imread(file_path)
    plt.imshow(img)
    plt.show()
