import sys
import numpy as np
import csv

def read_datas (file_name) :
	try :
		file = open(file_name, 'r')
	except Exception as exc:
		print("File error : {}".format(exc.__class__))
		exit(0)
	reader = csv.reader(file)
	return list(reader)

def normalize_minmax (arr_data):
	return (arr_data - arr_data.min()) / (arr_data.max() - arr_data.min())

def unnormalize_minmax (arr_data, arr_normalized_data):
	return arr_normalized_data * (arr_data.max() - arr_data.min()) + arr_data.min()

def estimatePrice(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)

def cost_fct(arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1])**2
	result = arr_errors.mean()
	return result

def drv_cost_fct_theta0 (arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1])
	return arr_errors.mean()

def drv_cost_fct_theta1 (arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1]) * arr_datas[:,0]
	return arr_errors.mean()

args = sys.argv
if len(args) != 2 :
	print("Arguments provided are inconsistents. Need to povide the path of a valid csv file.")
	exit(0)

file_name = args[-1]
data = read_datas (file_name)
del(data[0])

theta0 = 0
theta1 = 0
learningRate = 0.1
limit = 1000

print("Initial values :\ntheta0 = {}\ntheta1 = {}\nlearninRate = {}\ntraining_iterations = {}".format(
	theta0,
	theta1,
	learningRate,
	limit
))

arr_datas = np.array(data, dtype = 'i')

# Min Max Normalization
arr_mileage_normalized = normalize_minmax(arr_datas[:,0]).reshape((len(arr_datas[:,0])),1)
arr_price_normalized = normalize_minmax(arr_datas[:,1]).reshape((len(arr_datas[:,1]),1))
arr_normalized_datas = np.concatenate([arr_mileage_normalized,arr_price_normalized], axis = 1)

count = 0
while (count < limit) :
	# print(count)
	# print("fonction de cout : ", cost_fct(arr_normalized_datas, theta0,theta1))
	# print("drv en theta0 : ", drv_cost_fct_theta0 (arr_normalized_datas,theta0,theta1)) # facteur 2
	# print("drv en theta1 : ", drv_cost_fct_theta1 (arr_normalized_datas,theta0,theta1)) # facteur 2
	new_theta0 = theta0 - learningRate * drv_cost_fct_theta0 (arr_normalized_datas, theta0,theta1)
	new_theta1 = theta1 - learningRate * drv_cost_fct_theta1 (arr_normalized_datas , theta0, theta1)
	theta0 = new_theta0
	theta1 = new_theta1
	# print("nouveau theta 0 = ", theta0)
	# print("nouveau theta 1 = ", theta1)
	# print("\n")
	count+=1

arr_estimated_price = estimatePrice(arr_normalized_datas[:,0], theta0, theta1)
arr_estimated_price_unormalized = unnormalize_minmax(arr_datas[:,1] ,arr_estimated_price)

final_theta0 = unnormalize_minmax(arr_datas[:,1] ,estimatePrice(0, theta0, theta1))
final_theta1 = (unnormalize_minmax(arr_datas[:,1] ,estimatePrice(1, theta0, theta1)) - unnormalize_minmax(arr_datas[:,1] ,estimatePrice(0, theta0, theta1))) / unnormalize_minmax(arr_datas[:,0], 1)

# arr_estimated_price_final = estimatePrice(arr_normalized_datas[:,0], final_theta0, final_theta1)

with open("model_parameters.txt", 'w') as model_parameters_file :
	model_parameters_file.writelines([str(final_theta0), "\n", str(final_theta1)])

print("\nResults :\ntheta0 = {}\ntheta1= {}".format(final_theta0, final_theta1))
print(final_theta0 / final_theta1)

