import numpy as np

print("Hello")
revenus = [-2,1,-1,2]

print(revenus)

print(np.mean(revenus))
print(np.abs(revenus))
print(np.exp(revenus))


arr_revenus = np.array(revenus)

print(arr_revenus)

arr_zeros = np.zeros(14)
print(arr_zeros)

arr_ones = np.ones(14)
print(arr_ones)

arr_arranged = np.arange(12, 44, 2)
print(arr_arranged)

arr_linspaced = np.linspace(10,44, 5)
print(arr_linspaced)

print(arr_linspaced.dtype)

print(arr_linspaced[1])
print(arr_linspaced[-1])

arr_linspaced[-2] = 42.42

print(arr_linspaced)
