import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def uppgift5():
    def riemann(func, a, b, n):
        dx = (b-a)/n
        x = np.arange(a, b, step=dx)
        y = func(x)
        return y.sum()*dx

    def f(x):
        return x/np.cbrt(x**2+4)

    def g(x):
        return np.sqrt(x)*np.log(x)

    print(riemann(f, 0, 2, 100))
    print(riemann(g, 1, 4, 100))


def uppgift6():
    t = np.linspace(start=0, stop=10, num=100)
    plt.figure()
    for a in range(-4, 5):
        for b in range(-4, 5):
            q_a = np.power(np.e, -t) * (a*np.cos(t)+b *
                                        np.sin(t)) + np.cos(t) + 2*np.sin(t)
            plt.plot(t, q_a)
    q_b = np.cos(t)+2*np.sin(t)
    plt.plot(t, q_b, label='cos(t)+2sin(2)')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('q(t)')
    plt.ylim(-10, 10)
    plt.grid()
    plt.show()


def uppgift7():
    x = np.linspace(start=-10, stop=10, num=100)
    y_taylor = 0.2
    plt.figure()
    for k in range(14):
        y_taylor += (np.power((-1), k) * (np.power(x, 2*k+1) /
                     np.math.factorial(2*k+1))).astype('float64')
        if k == 13:
            plt.plot(x, y_taylor, label='Taylor (+0.2)')

    y_sin = np.sin(x)
    plt.plot(x, y_sin, label='sin(x)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-10, 10)
    plt.grid()
    plt.show()


def uppgift8():
    def y(x):
        return x-np.cos(x)

    def y_der(x):
        return 1+np.sin(x)

    x_plot = np.linspace(start=-5, stop=5)
    y1 = x_plot
    y2 = np.cos(x_plot)
    a = -6  # -5-cos(-5) = -7.0
    b = 2  # 2-cos(2) = 2.42
    c = 0
    it = 0
    while True:
        it += 1         # Kräver flera iterationer
        c = (a+b)/2
        if y(c) < 0:
            a = c
        else:
            b = c
        if np.abs(a-b) < 10**-12:
            print(c)
            print('it: ' + str(it))
            break

    x = a       # Når Pythons gräns
    it = 0
    while True:         # Mer rätt
        it += 1         # Kräver färre iterationer
        x_temp = x
        x -= y(x)/y_der(x)
        if np.abs(x-x_temp) < 10**-12:
            print(x)
            print('it: ' + str(it))
            break

    x_solve = fsolve(y, 0)
    print('Svar: ' + str(x_solve))

    plt.figure()
    plt.plot(x_plot, y1, label='y=x')
    plt.plot(x_plot, y2, label='y=cos(x)')

    # plt.plot(x, y3, label='y=x-cos(x)')
    plt.legend()
    plt.xlabel('x')
    plt.xlabel('y')
    plt.grid()
    plt.show()


def uppgift9():
    def cone():
        ax = plt.axes(projection='3d')
        r = np.linspace(0, 10, 100)
        v = np.linspace(0, 2*np.pi, 100)
        R, V = np.meshgrid(r, v)
        X = R * np.cos(V)
        Y = R * np.sin(V)

        Z = -np.sqrt(X**2 + Y**2)+10
        ax.plot_surface(X, Y, Z)

    def pyramid():
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        x1 = [0, 0, 3]
        y1 = [0, 6, 3]
        z1 = [0, 0, 6]

        x2 = [0, 6, 3]
        y2 = [0, 0, 3]
        z2 = [0, 0, 6]
        
        x3 = [6, 0, 3]
        y3 = [6, 6, 3]
        z3 = [0, 0, 6]
        
        x4 = [6, 6, 3]
        y4 = [6, 0, 3]
        z4 = [0, 0, 6]
        
        x5 = [0, 0, 6, 6]
        y5 = [0, 6, 6, 0]
        z5 = [0, 0, 0, 0]

        vertices1 = [list(zip(x1, y1, z1))]
        vertices2 = [list(zip(x2, y2, z2))]
        vertices3 = [list(zip(x3, y3, z3))]
        vertices4 = [list(zip(x4, y4, z4))]
        vertices5 = [list(zip(x5, y5, z5))]
            
        poly1 = Poly3DCollection(vertices1, alpha=0.8)
        poly2 = Poly3DCollection(vertices2, alpha=0.8)
        poly3 = Poly3DCollection(vertices3, alpha=0.8)
        poly4 = Poly3DCollection(vertices4, alpha=0.8)
        poly5 = Poly3DCollection(vertices5, alpha=0.8)

        ax.add_collection3d(poly1)
        ax.add_collection3d(poly2)
        ax.add_collection3d(poly3)
        ax.add_collection3d(poly4)
        ax.add_collection3d(poly5)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_zlim(0, 6)

    def half_sphere():
        ax = plt.axes(projection='3d')
        R = 10
        v = np.linspace(0, 2*np.pi, 100)
        u = np.linspace(0, np.pi/2, 100)
        V, U = np.meshgrid(v, u)
        X = R * np.sin(U) * np.cos(V)
        Y = R * np.sin(U) * np.sin(V)
        Z = R * np.cos(U)
        ax.plot_surface(X, Y, Z)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

    def spiral():
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        z = np.linspace(0, 15, 100)
        x1 = np.cos(z)
        y1 = np.sin(z)
        
        x2 = np.cos(z+np.pi)
        y2 = np.sin(z+np.pi)
        
        ax.plot3D(x1, y1, z, 'red')
        ax.plot3D(x2, y2, z, 'blue')

    # cone()
    # pyramid()
    # half_sphere()
    # spiral()
    plt.show()


val = int(input('Välj uppgiften du ska kolla på (int): '))
functions = [uppgift1, uppgift2, uppgift3, uppgift4,
             uppgift5, uppgift6, uppgift7, uppgift8,
             uppgift9]
functions[val-1]()
