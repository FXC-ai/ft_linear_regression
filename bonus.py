import sys
import matplotlib.pyplot as plt
import numpy as np

from libft_linear_regression import read_datas


def display_values(arr_mileage, arr_price, id_graph):
	fig = plt.figure(id_graph)
	plt.scatter(arr_mileage, arr_price, marker = 'P')
	plt.grid()
	plt.show()
	plt.close()

def display_model(arr_mileage,  arr_estimated_price, id_graph):
	fig = plt.figure(id_graph)
	plt.plot(arr_mileage, arr_estimated_price, c = "green")
	plt.grid()
	plt.show()
	plt.close()

args = sys.argv
if len(args) != 2 :
	print("Arguments provided are inconsistents. Need to povide the path of a valid csv file.")
	exit(0)

file_name = args[-1]
data = read_datas (file_name)
del(data[0])

arr_datas = np.array(data, dtype = 'i')

display_values(arr_datas[:,0], arr_datas[:,1], 1)
