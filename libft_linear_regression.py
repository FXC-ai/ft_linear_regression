import numpy as np
import csv
import matplotlib.pyplot as plt

def read_datas_to_array (file_name) :
	try :
		file = open(file_name, 'r')
	except Exception as exc:
		print("File error : {}".format(exc.__class__))
		exit(0)
	reader = csv.reader(file)
	datas = list(reader)
	del(datas[0])
	arr_datas = np.array(datas, dtype = 'i')
	return arr_datas

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

def read_model_parameters():
	with open("model_parameters.txt", "r") as model_parameters_file :
		model_parameters_file = open("model_parameters.txt", "r")
		list_str_parameters = model_parameters_file.readlines()
	theta0 = float(list_str_parameters[0])
	theta1 = float(list_str_parameters[1])
	return {"theta0" : theta0, "theta1" : theta1}

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

def display_values_and_model(arr_mileage, arr_price, arr_estimated_price, id_graph):
	fig = plt.figure(id_graph)
	plt.scatter(arr_mileage, arr_price, marker = 'P')
	plt.plot(arr_mileage, arr_estimated_price, c = "green")
	plt.grid()
	plt.show()
	plt.close()

arr_datas = read_datas_to_array("data.csv")