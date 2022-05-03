import numpy as np
import matplotlib.pyplot as plt

# UPPGIFT 1
def uppgift1():
    a = np.arange(1, 6, 1)

    b = np.arange(0, 2*np.pi, 0.1)

    c = np.empty(shape=(3, 2))
    c[0] = [1, 2]
    c[1] = [3, 4]
    c[2] = [5, 6]

    d = np.append(a, [6, 7])

    e = np.empty(shape=(2, 5))
    e[0] = a
    e[1] = np.arange(-1, -6, -1)

    f = []

    for i in range(b.size):
        # f = np.append(f, np.sin(b[i]))
        f = np.append(f, np.around(np.sin(b[i]), 4))

    print(a,'\n')
    print(b,'\n')
    print(c,'\n')
    print(d,'\n')
    print(e,'\n')
    print(f,'\n')

val = int(input('Välj uppgiften du ska kolla på (int): '))
functions = [uppgift1()]
functions[val-1]