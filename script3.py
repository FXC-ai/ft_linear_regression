import matplotlib.pyplot as plt
import numpy as np


datas = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],[1,3,5,5,3,7,8,9,10,-4,-4,-4,10,9,8,7,3,5,5,3,1]])

print(datas[0,:])
print(datas[1,:])

graph0 = plt.plot(datas[0,:],2*datas[1,:], marker='P')

plt.show()
