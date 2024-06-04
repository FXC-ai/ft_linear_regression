import numpy as np
import csv
import matplotlib.pyplot as plt

file = open("data.csv", 'r')
reader = csv.reader(file)
data = list(reader)
del(data[0])

nbr_data = len(data)

print("Nombre d'observations = {}\n".format(nbr_data))

arr_datas = np.array(data, dtype = 'i')
print("Données initiales : ")
print(arr_datas)
print("\n")

arr_sorted_datas = arr_datas[arr_datas[:,1].argsort()]
print("Données triées par ordre décroissant de kilométrage : ")
print(arr_datas)
print("\n")



def estimatePrice (x ,theta0, theta1) :
    return theta0 + (theta1 * x)


theta0 = 8400
theta1 = -0.02

estimatedPrice = estimatePrice(arr_sorted_datas[:,0], 8400, -0.020)
print("Valeurs estimées par le modèle pour theta0 = {} et theta1 = {} : ".format (8500, -0.02))
print(estimatedPrice)
print("\n")


def fonction_de_cout (arr_mileage, observationsPrice, theta0, theta1) :
    arr_error = (estimatePrice(arr_mileage, theta0, theta1) - observationsPrice)**2
    print("Erreurs :")
    print (arr_error)
    print("\n")
    return arr_error.mean()


mean_error = fonction_de_cout(arr_sorted_datas[:,0], arr_sorted_datas[:,1], theta0, theta1)

print("Moyenne des erreurs = {}".format(mean_error))

plt.figure(1)
plt.scatter(arr_sorted_datas[:,0], arr_sorted_datas[:,1], marker = 'P')
plt.plot(arr_sorted_datas[:,0], estimatedPrice, c = "green")
plt.grid()
plt.show()
plt.close()


arr_test_theta1 = np.arange(-0.5,0.5,0.005)
# print(arr_test_theta1)





