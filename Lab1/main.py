import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start=0, stop=10, num=100)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure()
plt.plot(x, y_sin, label='sin(x)')
plt.plot(x, y_cos, label='cos(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()