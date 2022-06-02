# import numpy as np
# import time
# import matplotlib.pyplot as plt

# x = np.linspace(0, 10, 100)
# y = np.cos(x)

# plt.ion()

# figure, ax = plt.subplots(figsize=(8,6))
# line1, = ax.plot(x, y)

# plt.title("Dynamic Plot of sinx",fontsize=25)

# plt.xlabel("X",fontsize=18)
# plt.ylabel("sinX",fontsize=18)

# for p in range(100):
#     updated_y = np.cos(x-0.05*p)
    
#     line1.set_xdata(x)
#     line1.set_ydata(updated_y)
    
#     figure.canvas.draw()
    
#     figure.canvas.flush_events()
#     time.sleep(0.1)

###############################################################################################
# import matplotlib.pyplot as plt
# import numpy as np

# plt.ion()
# fig, ax = plt.subplots()
# x, y = [],[]
# sc = ax.scatter(x,y)
# plt.xlim(0,10)
# plt.ylim(0,10)

# plt.draw()
# for i in range(1000):
#     x.append(np.random.rand(1)*10)
#     y.append(np.random.rand(1)*10)
#     sc.set_offsets(np.c_[x,y])
#     fig.canvas.draw_idle()
#     plt.pause(0.1)

# plt.waitforbuttonpress()


# import numpy as np 
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt



# def update_scat(h1,new_data):
# 	# xdata, ydata, zdata = hl._verts3d
# 	# hl.set_xdata(new_data[0])
# 	# hl.set_ydata(new_data[1])
# 	# hl.set_3d_properties(new_data[2])
#     h1.set_offsets(new_data)
#     plt.draw()


# spk = (np.random.rand(12,12,32)>0.8)*1.
# x,y,z = np.where(spk==1)

# map = plt.figure()
# map_ax = Axes3D(map)
# map_ax.autoscale(enable=True, axis='both', tight=True)

# # # # Setting the axes properties
# # map_ax.set_xlim3d([0.0, 10.0])
# # map_ax.set_ylim3d([0.0, 10.0])
# # map_ax.set_zlim3d([0.0, 10.0])

# hl= map_ax.scatter([0], [0], [0])

# update_scat(hl,[x,y,z])
# plt.show(block=False)
# plt.pause(1)

# spk = (np.random.rand(12,12,32)>0.8)*1.
# x,y,z = np.where(spk==1)
# update_scat(hl,[x,y,z])
# plt.show(block=False)
# plt.pause(2)

# spk = (np.random.rand(12,12,32)>0.8)*1.
# x,y,z = np.where(spk==1)

# update_scat(hl,[x,y,z])
# plt.show(block=True)


import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import gridspec

spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[2, 1])

def update_scat(map,new_data):
    # ax = Axes3D(map)
    
    ax = map.add_subplot(1, 2, 1, projection='3d')
    # ax.set_axis_off()
    ax.scatter(new_data[0][0],new_data[0][1],new_data[0][2],s=2,m='.')

    ax1 = map.add_subplot(1, 2, 2, projection='3d')
    ax1.set_axis_off()
    ax1.scatter(new_data[1][0],new_data[1][1],new_data[1][2],s=2,m='.')

    ax.title.set_text('Layer 1')
    ax1.title.set_text('Layer 2')

    ax.set_xlabel('w')
    ax.set_ylabel('h')
    ax.set_zlabel('c')

    ax1.set_xlabel('w')
    ax1.set_ylabel('h')
    ax1.set_zlabel('c')

    map.canvas.draw()

def update_scat(map,new_data):
    # ax = Axes3D(map)
    
    ax = map.add_subplot(spec[0], projection='3d')
    # ax.set_axis_off()
    ax.scatter(new_data[0][0],new_data[0][1],new_data[0][2],s=2)

    ax1 = map.add_subplot(spec[1], projection='3d')
    ax1.set_axis_off()
    ax1.scatter(new_data[1][0],new_data[1][1],new_data[1][2],s=2)

    ax.title.set_text('Layer 1')
    ax1.title.set_text('Layer 2')

    ax.set_xlabel('w')
    ax.set_ylabel('h')
    ax.set_zlabel('c')

    ax1.set_xlabel('w')
    ax1.set_ylabel('h')
    ax1.set_zlabel('c')

    map.canvas.draw()

spk = (np.random.rand(12,12,32)>0.8)*1.
x,y,z = np.where(spk==1)

map = plt.figure(figsize=(10,5))

for i in range(10):
    spk = (np.random.rand(12,12,256)>0.8)*1.
    x1,y1,z1 = np.where(spk==1)

    spk = (np.random.rand(32,32,32)>0.9)*1.
    x,y,z = np.where(spk==1)

    update_scat(map,[[x,y,z],[x1,y1,z1]])
    # plt.show(block=False)
    plt.pause(1) 


plt.show(block=True)

