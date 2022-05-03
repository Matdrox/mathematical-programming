import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


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

    print(a, '\n')
    print(b, '\n')
    print(c, '\n')
    print(d, '\n')
    print(e, '\n')
    print(f, '\n')


def uppgift2():
    def function_a():
        x = 3
        return x*x

    def function_b1():
        x = np.array([2, 4, 5])
        result = []
        for i in range(x.size):
            result = np.append(result, np.square(x[i]))
        return result

    def function_b2():
        x = np.array([2, 4, 5])
        return np.dot(x, x)

    def function_c1():
        x = np.array([[2, 4, 5],
                      [3, 4, 6],
                      [1, 5, 7]])
        result = np.empty_like(x)
        rows, cols = x.shape
        for i in range(rows):
            for j in range(cols):
                result[i, j] = np.square(x[i, j])
        return result

    def function_c2():
        x = np.array([[2, 4, 5],
                      [3, 4, 6],
                      [1, 5, 7]])
        return np.matmul(x, x)

    print(function_a(), '\n')
    print(function_b1(), '\n')
    print(function_b2(), '\n')
    print(function_c1(), '\n')
    print(function_c2(), '\n')


def uppgift3():
    x = np.linspace(start=-10, stop=10, num=100)
    y = 1 + x + 4/np.square(x-2)
    asymptote_y = x + 1
    plt.figure()
    plt.plot(x, y, label='f(x) = 1 + x + 4/np.square(x-2)')
    plt.axvline(x=2, label='x = 2', color='darkorange')
    plt.plot(x, asymptote_y, label='f(x) = x + 1', color='darkorange')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim(-10, 10)
    plt.grid()
    plt.show()


def uppgift4():
    def f(x):
        return 1 + np.sin(x) + 0.5*np.cos(4*x)
    h = 0.001
    x = np.linspace(start=0, stop=6, num=100)
    y_numerical = (f(x+h) - f(x))/h
    y_analytical = np.cos(x) - 2*np.sin(4*x)+0.5
    plt.figure()
    plt.plot(x, y_analytical, label='Analytical + 0.5')
    plt.plot(x, y_numerical, label='Numerical')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim(-10, 10)
    plt.grid()
    plt.show()


val = int(input('Välj uppgiften du ska kolla på (int): '))
functions = [uppgift1, uppgift2, uppgift3, uppgift4]
functions[val-1]()
