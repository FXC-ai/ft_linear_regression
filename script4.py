import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def estimatePrice(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)

def cost_fct(theta0, theta1):
	return 750 * theta1 * theta1 + 50 * theta1 * theta0 + theta0 * theta0 - 3355 * theta1 - 113.5 * theta0 + 3757.25

def cost_fct_2(arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1])**2
	result = arr_errors.mean()
	return result

def drv_cost_fct_theta0 (theta0, theta1):
	return 50 * theta1 + 2 * theta0 - 113.5

def drv_cost_fct_theta0_2 (arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1])
	return arr_errors.mean()

def drv_cost_fct_theta1 (theta0, theta1):
	return 1500 * theta1 + 50 * theta0 - 3355

def drv_cost_fct_theta1_2 (arr_datas, theta0, theta1):
	arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
	arr_errors = (arr_estimatedPrice - arr_datas[:,1]) * arr_datas[:,0]
	return arr_errors.mean()

def display_cost_fct():
	fig = plt.figure(1)
	ax = plt.axes(projection="3d")
	arr_theta0 = np.linspace(0, 14, 10)
	arr_theta1 = np.linspace(0, 4, 10)
	values = np.zeros((len(arr_theta0), len(arr_theta1)))
	for i in range(len(arr_theta0)):
		for j in range (len(arr_theta1)):
			values[i,j] = cost_fct(arr_theta0[i], arr_theta1[j])

	ax.plot_surface(arr_theta0, arr_theta1, values, cmap="viridis", edgecolor="green")
	plt.show()
	plt.close()


def display_model(arr_datas, arr_estimatedPrice):
	fig = plt.figure(2)
	plt.scatter(arr_datas[:,0], arr_datas[:,1], marker = 'P')
	plt.plot(arr_datas[:,0], arr_estimatedPrice, c = "green")
	plt.grid()
	plt.show()
	plt.close()

# def display_drv_cost_fct_theta0 ():
# 	fig = plt.figure()
# 	arr_theta0 = np.linspace(0, 14, 10)
# 	values = 
# 	plt.plot()

arr_datas = np.array([[10,20,30,40],[25,48,66,88]]).T
theta0 = 0
theta1 = 0

# print(cost_fct(theta0,theta1), cost_fct_2(arr_datas, theta0,theta1))
# print("drv en theta0 : ", drv_cost_fct_theta0 (theta0,theta1), 2*drv_cost_fct_theta0_2 (arr_datas, theta0,theta1))
# print("drv en theta1 : ", drv_cost_fct_theta1 (theta0,theta1), 2*drv_cost_fct_theta1_2 (arr_datas, theta0,theta1))


learningRate_theta_0 = -0.1
learningRate_theta_1 = -0.001
count = 0
limit = 10000

while (count < limit) :
	print("fonction de cout : ", cost_fct(theta0,theta1), cost_fct_2(arr_datas, theta0,theta1))
	print("drv en theta0 : ", drv_cost_fct_theta0 (theta0,theta1), 2*drv_cost_fct_theta0_2 (arr_datas, theta0,theta1))
	print("drv en theta1 : ", drv_cost_fct_theta1 (theta0,theta1), 2*drv_cost_fct_theta1_2 (arr_datas, theta0,theta1))
	new_theta0 = theta0 + learningRate_theta_0 * drv_cost_fct_theta0_2 (arr_datas, theta0,theta1)
	new_theta1 = theta1 + learningRate_theta_1 * drv_cost_fct_theta1_2 (arr_datas, theta0, theta1)
	theta0 = new_theta0
	theta1 = new_theta1
	print("nouveau theta 0 = ", theta0)
	print("nouveau theta 1 = ", theta1)
	print("\n")
	count+=1



display_cost_fct()

arr_estimatedPrice = estimatePrice(arr_datas[:,0], theta0, theta1)
display_model(arr_datas, arr_estimatedPrice)

