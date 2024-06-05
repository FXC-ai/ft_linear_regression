import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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


# Calcule pour un kilometrage donné le prix de la voiture
def estimatePrice (x ,theta0, theta1) :
    return theta0 + (theta1 * x)

# Calcule pour tous le kilometrages donnés le prix estimé
def calculate_model(arr_mileage, theta0, theta1):
    estimatedPrices = estimatePrice(arr_mileage, theta0, theta1)
    # print("Prix estimés par le modèle pour theta0 = {} et theta1 = {} : ".format (theta0, theta1))
    # print(estimatedPrices)
    # print("\n")
    return estimatedPrices

# Evalue la performance du modele
def fonction_de_cout (estimatedPrices, observationsPrice) :
    arr_error_square = (estimatedPrices - observationsPrice)**2
    arr_error_square_mean = arr_error_square.mean()
    # print("Valeur de la fonction de cout = {}".format(arr_error_square_mean))
    return arr_error_square_mean


def calculate_random_gradient_descente (arr_mileage, arr_observations) :
    arr_theta0 = np.linspace(0, 12000, 10)
    arr_theta1 = np.linspace(-0.5, 0.5, 10)
    results = np.zeros((len(arr_theta0), len(arr_theta1)))
    for i in range(len(arr_theta0)):
        for j in range (len(arr_theta1)):
            results[i,j] = fonction_de_cout(calculate_model(arr_mileage, arr_theta0[i], arr_theta1[j]), arr_observations)

    return results
    # return fonction_de_cout(mileage)

def display_cost_fct (arr_performances):
    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    arr_theta0 = np.linspace(0, 10000, 10)
    arr_theta1 = np.linspace(-0.5, 0.5, 10)
    ax.plot_surface(arr_theta0, arr_theta1, arr_performances, cmap="viridis", edgecolor="green")
    plt.show()

arr_mileage = arr_sorted_datas[:,0]
arr_observations = arr_sorted_datas[:,1]

r = calculate_random_gradient_descente (arr_mileage, arr_observations)

print(r)

display_cost_fct(r)

exit(13)
# arr_theta0 = np.linspace(0, 10000, 100)
# arr_theta1 = np.linspace(-0.5, 0.5, 0.01)
# performance = arr_theta0 * arr_theta1
# print(performance)




# Calcule la dérivée de la fonction de cout selon theta0
def derivative_cost_fct_theta0 (estimatedPrice, observationsPrice) :
    arr_error = (estimatedPrice - observationsPrice)
    derivative_theta0 = arr_error.mean()
    print("derivée en theta0 = {}".format(derivative_theta0))
    return derivative_theta0

# Calcule la dérivée de la fonction de cout selon theta1
def derivative_cost_fct_theta1 (mileage, estimatedPrice, observationsPrice) :
    arr_error = (estimatedPrice - observationsPrice) * mileage
    derivative_theta1 = arr_error.mean()
    print("derivée en theta1 = {}".format(derivative_theta1))
    return derivative_theta1

# Calcule du nouveau theta selon le learningRate choisis
def calculate_new_theta (derivativeValue, learningRate):
    print("nouveau theta = {}".format(derivativeValue * learningRate))
    return derivativeValue * learningRate

def display_model(mileage, observationsPrice, estimatedPrice):
    plt.figure(1)
    plt.scatter(mileage, observationsPrice, marker = 'P')
    plt.plot(mileage, estimatedPrice, c = "green")
    plt.grid()
    plt.show()
    plt.close()



learningRate = 0.0000001
theta0 = 0
theta1 = 0
mileage = arr_sorted_datas[:,0]
observationsPrice = arr_sorted_datas[:,1]

display_cost_fct(mileage, observationsPrice)


limit = 0
count = 0
while count < limit :
    estimatedPrice = calculate_model(mileage, theta0, theta1)
    display_model(mileage, observationsPrice, estimatedPrice)
    mean_error = fonction_de_cout(mileage, theta0, theta1, observationsPrice)
    theta0 = calculate_new_theta(derivative_cost_fct_theta0(estimatedPrice, observationsPrice), learningRate)
    theta1 = calculate_new_theta(derivative_cost_fct_theta1(mileage, estimatedPrice, observationsPrice), learningRate)
    print("Nouvelles valeurs de theta0 = {} et theta1 = {}".format(theta0, theta1))
    count += 1




# arr_test_theta1 = np.arange(-0.5,0.5,0.005)
# print(arr_test_theta1)





