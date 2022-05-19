import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import io
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d
from PIL import Image


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
        plt.imshow(M.T, cmap='gray', extent=(a_min, a_max, b_min, b_max))
        plt.show()

    full()
    # zoom()


def uppgift2():
    img = plt.imread('borggarden_small.jpg')
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_grayscale = R * 0.2989 + G * 0.5870 + B * 0.1140

    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    xd = ndimage.convolve(img_grayscale, Gx, mode='constant')
    yd = ndimage.convolve(img_grayscale, Gy, mode='constant')

    rows, cols = img_grayscale.shape

    S = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            S[i][j] = np.sqrt(np.square(xd[i][j]) + np.square(yd[i][j]))

    S[S > 250] = 0
    S[S < 50] = 255

    n = 1
    M = np.ones((n, n))
    result = ndimage.convolve(S, M, mode='constant')

    result[result > 255*n*n] = 0
    result[result < 255*n*n] = 0

    plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    plt.show()


def uppgift3():
    # samplerate, data = io.wavfile.read('Piano_1_C.wav')
    # length = data.shape[0]/samplerate
    # time = np.linspace(0, length, data.shape[0])
    # plt.plot(time, data[:, 0], label='Left Channel')
    # plt.plot(time, data[:, 1], label='Right Channel')
    # plt.legend()
    # plt.title('Sine Wave')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    def a():
        fs, data = io.wavfile.read('Piano_1_C.wav')  # Samplerate: 44100
        audio = data[0:fs]
        F = fft(audio)
        fourier_half = len(F)/2

        # Python tror att det är en float... lägger till .0
        plt.plot(abs(F[:(int(fourier_half))]))
        plt.xlabel('Tid')
        plt.ylabel('Amplitud')
        plt.title('Piano Wave')
        plt.xlim([0, 3000])

    def b():
        def freq_to_note(f):
            notes = ['A', 'A#', 'B', 'C', 'C#', 'D',
                     'D#', 'E', 'F', 'F#', 'G', 'G#']
            num = 12*np.log2(f/440) + 49
            num = round(num)
            note = (num-1) % len(notes)
            note = notes[note]

            return note

        fs, data = io.wavfile.read('Piano_5.wav')
        audio = data[0:fs]
        F = fft(audio)
        freq = np.argmax(abs(F))

        print(f'Frekvensen {freq} är tonen {freq_to_note(21)}.')

    def c():
        notes_e = [41, 82, 165, 330, 659, 1319, 2637]
        notes_ess = [39, 78, 156, 311, 622, 1245, 2489]
        offset = 5

        fs, data = io.wavfile.read('Cdur.wav')
        audio = data[0:fs]
        F = fft(audio)
        fourier_half = abs(F[:int(len(F))])
        plt.plot(fourier_half, label='C-dur', color='darkorange')

        for i in range(len(notes_e)):
            temp = fourier_half[notes_e[i]-offset: notes_e[i]+offset]
            fourier_half[notes_ess[i]-offset: notes_ess[i]+offset] = temp
            fourier_half[notes_e[i]-offset:notes_e[i]+offset] = 0

        for i in range(len(notes_e)):
            temp = fourier_half[fourier_half.size - notes_e[i] -
                                offset: fourier_half.size - notes_e[i]+offset]
            fourier_half[fourier_half.size - notes_ess[i] -
                         offset: fourier_half.size - notes_ess[i]+offset] = temp
            fourier_half[fourier_half.size - notes_e[i] -
                         offset:fourier_half.size - notes_e[i]+offset] = 0

        fourier_half[0:500] = 0
        fourier_half[42000:44100] = 0

        plt.plot(fourier_half, label='C-moll', color='darkblue')
        plt.legend()
        plt.xlabel('Tid')
        plt.ylabel('Amplitud')
        plt.title('Piano Wave')
        plt.xlim([40000, 44100])

        data = ifft(fourier_half)
        io.wavfile.write('Cmoll.wav', 44100, data.astype(np.int16))

    c()
    plt.show()


val = int(input('Välj uppgiften du ska kolla på (int): '))
functions = [uppgift1, uppgift2, uppgift3]
functions[val-1]()
