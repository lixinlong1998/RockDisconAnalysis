import numpy as np

# x = -0.0714398
# y = 0.976184

x = 0.119061
y = -0.309555


d = np.arctan(x / y)
print('dip dir:', 180 + d * 180 / np.pi)

l = np.sqrt(x * x + y * y)
print('dip:', l * 90)
