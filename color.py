from matplotlib import pyplot as plt
from random import random

x = list(range(10))
y = [random() for i in range(10)]
z = [random() for i in range(10)]
plt.scatter(x,y,c=z,cmap="Blues")
plt.colorbar()
plt.show()