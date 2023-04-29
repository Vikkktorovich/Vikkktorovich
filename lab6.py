import numpy as np
import cv2
from matplotlib import pyplot as plt

A = np.array([[-1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
B = np.array([[1., -2., -3.], [7., 8., 9.], [4., 5., 6.], ])

C = A + B
D = A - B
E = A * B
F = A / B
G = A ** B

print('+\n', C, '\n')
print('-\n', D, '\n')
print('*\n', E, '\n')
print('/\n', F, '\n')
print('**\n', G, '\n')













# скаляры
A = 2
B = 3

print(np.dot(A, B), '\n')

# вектор и скаляр
A = np.array([2., 3., 4.])
B = 3

print(np.dot(A, B), '\n')

# вектора
A = np.array([2., 3., 4.])
B = np.array([-2., 1., -1.])

print(np.dot(A, B), '\n')

# тензор и скаляр
A = np.array([[2., 3., 4.], [5., 6., 7.]])
B = 2

print(np.dot(A, B), '\n')








A = np.random.rand(4, 5)

print('A\n', A, '\n')

print('min\n', np.min(A, 0), '\n')
print('max\n', np.max(A, 0), '\n')
print('mean\n', np.mean(A, 0), '\n')
print('average\n', np.average(A, 0), '\n')



I = cv2.imread('Stephen.jpeg')[:, :, ::-1]
plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(I)
plt.show()



