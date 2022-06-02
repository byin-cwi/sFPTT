import matplotlib.pyplot as plt
import numpy as np 

# x = np.linspace(0, 10, 100)
# fig = plt.figure()
# output_spk = np.zeros((100,50)).tolist()
# for p in range(50):
#     spk = (np.random.rand(50)>0.8)*1.

#     output_spk.pop(0)
#     output_spk.append(spk)

    
#     plt.imshow(np.array(output_spk).T,cmap='gray')
#     plt.xlabel('Time (s)')
#     plt.ylabel('id')
#     plt.draw()  
#     plt.pause(0.2)
#     fig.clear()


import matplotlib.gridspec as gridspec
fig = plt.figure()
gs = gridspec.GridSpec(3,4)
ax0 = plt.subplot(gs[:2,:2])
ax1 = plt.subplot(gs[:2,2])
ax2 = plt.subplot(gs[:2,3])
ax3 = plt.subplot(gs[2,:])

output_spk = np.zeros((300,50)).tolist()
for p in range(50):
    spk = (np.random.rand(50)>0.8)*1.

    output_spk.pop(0)
    output_spk.append(spk)

    
    ax3.imshow(np.array(output_spk).T,cmap='gray')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('id')
    plt.draw()  
    plt.pause(0.002)
    # fig.clear()

