import sys
import matplotlib.pyplot as plt
import numpy as np

from libft_linear_regression import *



args = sys.argv
if len(args) != 2 :
	print("Arguments provided are inconsistents. Need to povide the path of a valid csv file.")
	exit(0)

file_name = args[-1]
data = read_datas (file_name)

arr_datas = np.array(data, dtype = 'i')
display_values(arr_datas[:,0], arr_datas[:,1], 1)

dict_params = read_model_parameters()
theta0 = dict_params["theta0"]
theta1 = dict_params["theta1"]

arr_estimated_price = theta0 + theta1 * arr_datas[:,0]

display_model (arr_datas[:,0], arr_estimated_price, 2)

display_values_and_model(arr_datas[:,0],arr_datas[:,1],arr_estimated_price, 3)
