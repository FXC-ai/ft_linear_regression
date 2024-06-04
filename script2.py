import numpy as np

tab1 = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(tab1)

print("\n")

print(np.ones((3,5)))

print("\n")

print(np.zeros((4,4)))

print("\n")

print(np.random.random(size=(4,4)))

print("\n")

print(np.random.randint(1, 10, size=(3, 3)))

matrice_0 = np.array([[1,2,3],[4,5,6],[7,8,9]])
matrice_1 = np.array([[1,2,3],[4,5,6],[7,8,9]])

matrice_2 = matrice_0 + matrice_1
print("\n")
print(matrice_2)

matrice_3 = matrice_0 * matrice_1
print("\n")
print(matrice_3)

print("\n")


matrice_4 = np.array([[2,-2],[5,3]])
matrice_5 = np.array([[-1,4],[7,-6]])

print(matrice_4)
print("\n")

print(matrice_5)


matrice_6 = np.dot(matrice_4, matrice_5)

print("\n")
print(matrice_6)

matrice_7 = np.array([[0,33,2, 1988],[1, 36, 0, 1991],[2, 73, 3, 1951],[3, 64, 0, 1959]])

matrice_7 = np.vstack((matrice_7, [4, 5, 0, 2019]))

print("\nmatrice_7")
print(matrice_7)

print("\n")
print(matrice_7[0,0])
print(matrice_7[0,:])
print(matrice_7[:,0])
print(matrice_7[1:-1,0])


matrice_8 = matrice_7[matrice_7[:,1] < 36]

print("\n")
print(matrice_8)

a = np.linspace(5, 10, 11)

print(a[a % 2 == 0])

b = np.array([[[1, 2],[4, 5]],[[6, 7],[8, 9]],[[10, 11],[12, 13]]])

print("\n")
print(b[2, :, :])