import sys
import numpy as np

from libft_linear_regression import *

args = sys.argv
if len(args) != 3 :
	print("Please enter valid args : python3 ft_linear_regression.py [file name].csv [flag bonus 0 or 1]")
	exit(0)

file_name = args[1]
flag = int(args[-1])

arr_datas = read_datas_to_array(file_name)

theta0 = 0
theta1 = 0
learningRate = 0.1
limit = 1300

print("Initial values :\ntheta0 = {}\ntheta1 = {}\nlearningRate = {}\ntraining_iterations = {}".format(
	theta0,
	theta1,
	learningRate,
	limit
))

# Min Max Normalization
arr_mileage_normalized = normalize_minmax_arr(arr_datas[:,0]).reshape((len(arr_datas[:,0])),1)
arr_price_normalized = normalize_minmax_arr(arr_datas[:,1]).reshape((len(arr_datas[:,1]),1))
arr_normalized_datas = np.concatenate([arr_mileage_normalized,arr_price_normalized], axis = 1)

count = 0
while (count < limit) :
	if flag == 1 and count % 10 == 0:
		cost = cost_fct(arr_normalized_datas, theta0,theta1)
		print(count, " | Fonction de cout : ", cost)
	tmp_theta0 = theta0 - learningRate * drv_cost_fct_theta0 (arr_normalized_datas, theta0,theta1)
	tmp_theta1 = theta1 - learningRate * drv_cost_fct_theta1 (arr_normalized_datas , theta0, theta1)
	theta0 = tmp_theta0
	theta1 = tmp_theta1
	count+=1

print("\nLinear regression : OK")
print("Normalized theta0 = {}\nNormalized theta1 = {}".format(theta0, theta1))

arr_estimated_price = estimatePrice(arr_normalized_datas[:,0], theta0, theta1)
arr_estimated_price_unormalized = unnormalize_minmax_arr(arr_datas[:,1] ,arr_estimated_price)

estimated_norm_price_max = estimatePrice(1, theta0, theta1)
estimated_norm_price_min = estimatePrice(0, theta0, theta1)

final_theta1 = (unnormalize_minmax(estimated_norm_price_max, arr_datas[:,1]) - unnormalize_minmax(estimated_norm_price_min, arr_datas[:,1])) / (arr_datas[:,0].max() - arr_datas[:,0].min())
final_theta0 = unnormalize_minmax(estimated_norm_price_min, arr_datas[:,1]) - final_theta1 * arr_datas[:,0].min()

with open("model_parameters.txt", 'w') as model_parameters_file :
	model_parameters_file.writelines([str(final_theta0), "\n", str(final_theta1)])

print("theta0 = ", final_theta0)
print("theta1 = ", final_theta1)

if flag == 1 :
	print("Fonction de coût = ", cost)
	display_values(arr_datas[:,0], arr_datas[:,1], 2)
	display_model(arr_datas[:,0], arr_estimated_price_unormalized,3)
	display_values_and_model(arr_mileage_normalized, arr_price_normalized, arr_estimated_price, 1)
	display_values_and_model(arr_datas[:,0], arr_datas[:,1], arr_estimated_price_unormalized, 4)
	display_cost_fct(arr_normalized_datas)
