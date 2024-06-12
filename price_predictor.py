import sys
import libft_linear_regression as lr

args = sys.argv
if len(args) != 2 or int(args[-1]) < 0 or int(args[-1]) > 1000000:
	print("Arguments provided are inconsistents. Please enter a number between 0 and 1000000.")
	exit(0)

mileage = int(args[-1])

# mileage = lr.normalize_minmax(mileage, lr.arr_datas[:,0])

dict_params = lr.read_model_parameters()
theta0 = dict_params["theta0"]
theta1 = dict_params["theta1"]

print("Mes parametres : " ,theta0, theta1)

estimated_price = theta0 + theta1 * mileage

estimated_price = 0 if estimated_price < 0 else estimated_price

print("The estimated price of the model for a mileage of {} is : {} ".format(mileage, estimated_price))