import numpy as np
import csv
import matplotlib.pyplot as plt

file = open("data.csv", 'r')

reader = csv.reader(file)

data = list(reader)

del(data[0])

arr_datas = np.array(data, dtype = 'i')

arr_sorted_datas = arr_datas[arr_datas[:,0].argsort()]

print(arr_datas)
print("\n")

def estimatePrice (x ,theta0, theta1) :
    return theta0 + (theta1 * x)

res = estimatePrice(arr_sorted_datas[:,0], 8000, -0.020)

plt.figure(1)
plt.scatter(arr_sorted_datas[:,0], arr_sorted_datas[:,1], marker = 'P')
plt.plot(arr_sorted_datas[:,0], res, c = "green")
plt.grid()
plt.show()
plt.close()

