import matplotlib.pyplot as plt
import numpy as np 

x = np.linspace(0, 10, 100)
fig = plt.figure()
output_spk = np.zeros((100,50)).tolist()
for p in range(50):
    spk = (np.random.rand(50)>0.8)*1.

    output_spk.pop(0)
    output_spk.append(spk)

    
    plt.imshow(np.array(output_spk).T,cmap='gray')
    plt.xlabel('Time (s)')
    plt.ylabel('id')
    plt.draw()  
    plt.pause(0.2)
    fig.clear()