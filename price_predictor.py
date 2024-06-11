import sys

args = sys.argv
if len(args) != 2 or int(args[-1]) < 0 or int(args[-1]) > 1000000:
	print("Arguments provided are inconsistents. Please enter a number between 0 and 1000000.")
	exit(0)

mileage = int(args[-1])

model_parameters_file = open("model_parameters.txt", "r")
list_str_parameters = model_parameters_file.readlines()

theta0 = float(list_str_parameters[0])
theta1 = float(list_str_parameters[1])

model_parameters_file.close()

estimated_price = theta0 + theta1 * mileage

estimated_price = 0 if estimated_price < 0 else estimated_price

print("The estimated price of the model for a mileage of {} is : {} ".format(mileage, estimated_price))