import numpy as np

print("Hello")
revenus = [-2,1,-1,2]

print("revenus", revenus)

print("np.mean(revenus)", np.mean(revenus))
print("np.abs(revenus)", np.abs(revenus))
print("np.exp(revenus)", np.exp(revenus))


print("\n")
arr_revenus = np.array(revenus)

print("arr_revenus", arr_revenus)

arr_zeros = np.zeros(14)
print("rr_zeros", arr_zeros)

arr_ones = np.ones(14)
print("arr_ones", arr_ones)

arr_arranged = np.arange(12, 44, 2)
print("arr_arranged", arr_arranged)

arr_linspaced = np.linspace(10,447, 21)
print("rr_linspaced", arr_linspaced)

print("arr_linspaced.dtype", arr_linspaced.dtype)

print("arr_linspaced[1] ", arr_linspaced[1])
print("arr_linspaced[-1]", arr_linspaced[-1])

arr_linspaced[-2] = 42.42
print("\n")

print("arr_linspaced", arr_linspaced)
print("\n")

print("arr_linspaced[7:14:3]", arr_linspaced[7:14:3])
print("\n")

print("arr_linspaced[:14:3]", arr_linspaced[:14:3])
print("\n")


print(arr_linspaced[::-1])
print(arr_linspaced[::2])
print("\n")


print(arr_linspaced[(arr_linspaced > 100) & (arr_linspaced < 200)])

print(arr_linspaced.shape)

print(arr_linspaced.mean(), arr_linspaced.min(), arr_linspaced.max(), arr_linspaced.argmin(), arr_linspaced.argmax())


print(arr_linspaced.sum())