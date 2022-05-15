import numpy as np
import matplotlib.pyplot as plt


def uppgift1():
    def mandelbrot(c):
        # z = 0
        # for _ in range(101):
        #     z = z*z + c
        #     if not np.abs(z) > 2:
        #         return True
        z = 0
        temp = 0
        while np.abs(z) < 2:
            z = z*z + c
            if temp == 100:
                return True
            # elif temp == 100 and abs(z) < 2:
            #     break
            temp += 1


    # a = np.arange(-2, 2.01, 0.01)
    # b = np.arange(-2, 2.01, 0.01)
    M = np.zeros(shape=(401, 401))

    def full():
        a_min = -2
        b_min = -2
        a_max = 2
        b_max = 2
        a = np.arange(a_min, a_max, 0.01)
        b = np.arange(b_min, b_max, 0.01)

        for i in a:
            for j in b:
                temp = mandelbrot(complex(i, j))
                if temp:
                    M[int((i+2)*100), int((j+2)*100)] = 1
        
        plt.imshow(M.T, cmap='gray', extent=(a_min, a_max, b_min, b_max))
        plt.show()

    def zoom():
        a_min = -1
        b_min = 0
        a_max = -0.6
        b_max = 0.4
        a = np.arange(a_min, a_max, 0.001)
        b = np.arange(b_min, b_max, 0.001)

        for i in a:
            for j in b:
                temp = mandelbrot(complex(i, j))
                if temp:
                    M[int((i+1)*1000), int((j)*1000)] = 1
        plt.imshow(N.T, cmap='gray', extent=(a_min, a_max, b_min, b_max))
        plt.show()

    full()
    # zoom()


val = int(input('Välj uppgiften du ska kolla på (int): '))
functions = [uppgift1]
functions[val-1]()
