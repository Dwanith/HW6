import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2



#Q1: wave plot

A = 325
f = 50
dt = 0.0002

t = np.arange(0, 0.02, 0.0001)
x_t = A * np.sin(2 * np.pi * f * t)

n = np.arange(0, 51)
x_n = A * np.sin(2 * np.pi * f * n * dt)

plt.figure(figsize=(10, 6))

plt.plot(t, x_t, label="Continuous Signal (x(t))", color="blue")

plt.stem(n * dt, x_n, linefmt='r-', markerfmt='ro', basefmt=" ", label="Discrete Signal (x[n])")

plt.title("Continuous and Discrete Signals")
plt.xlabel("Time (t) or Sample Index (n)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()


#Q2



t_q2 = np.linspace(-0.02, 0.05, 1000)
x_q2 = np.exp(1j * 100 * np.pi * t_q2)

real_part_q2 = x_q2.real
imag_part_q2 = x_q2.imag

plt.figure(figsize=(10, 6))
plt.plot(t_q2, real_part_q2, label="Real Part (cos)", color="blue")
plt.plot(t_q2, imag_part_q2, label="Imaginary Part (sin)", color="red", linestyle="--")
plt.title("Real and Imaginary Parts of $e^{j 100 \\pi t}$")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()



#Q3


N = 32
n = np.arange(N)

f_original = np.zeros(N)
f_original[5], f_original[10], f_original[15] = 1.5, 2, 1

f_translated = np.roll(f_original, 8)
f_mirrored = f_original[::-1]

F_original = np.fft.fft(f_original) / N
F_translated = np.fft.fft(f_translated) / N
F_mirrored = np.fft.fft(f_mirrored) / N

F_original_mag = np.abs(F_original)
F_translated_mag = np.abs(F_translated)
F_mirrored_mag = np.abs(F_mirrored)

plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.stem(n, f_original)
plt.title("Original Signal")

plt.subplot(3, 3, 2)
plt.stem(n, F_original.real, label="Real")
plt.stem(n, F_original.imag, label="Imaginary")
plt.title("F[u] (Original)")
plt.legend()

plt.subplot(3, 3, 3)
plt.stem(n, F_original_mag)
plt.title("|F[u]| (Original)")

plt.subplot(3, 3, 4)
plt.stem(n, f_translated)
plt.title("Translated Signal")

plt.subplot(3, 3, 5)
plt.stem(n, F_translated.real, label="Real")
plt.stem(n, F_translated.imag, label="Imaginary")
plt.title("F[u] (Translated)")
plt.legend()

plt.subplot(3, 3, 6)
plt.stem(n, F_translated_mag)
plt.title("|F[u]| (Translated)")

plt.subplot(3, 3, 7)
plt.stem(n, f_mirrored)
plt.title("Mirrored Signal")

plt.subplot(3, 3, 8)
plt.stem(n, F_mirrored.real, label="Real")
plt.stem(n, F_mirrored.imag, label="Imaginary")
plt.title("F[u] (Mirrored)")
plt.legend()

plt.subplot(3, 3, 9)
plt.stem(n, F_mirrored_mag)
plt.title("|F[u]| (Mirrored)")

plt.tight_layout()
plt.show()





#Q4: plot FT 50Hz


t = np.linspace(0, 1, 1000, endpoint=False)
f_t = 1 / (1 + 0.99 * np.cos(2 * np.pi * t))

F_f_t = np.fft.fft(f_t)
frequencies = np.fft.fftfreq(len(t), d=(t[1] - t[0]))

mask = np.abs(frequencies) < 50
frequencies = frequencies[mask]
F_f_t = F_f_t[mask]

plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(F_f_t), label="Magnitude of FFT")
plt.title("Fourier Transform of $f(t) = \\frac{1}{1 + 0.99 \\cos(2\\pi t)}$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()
plt.show()







#Q5: Fourier transform in real and fourier domian plotted


N, M = 20, 20
frequencies = [(2, 0), (3, 1), (7, -5)]

plt.figure(figsize=(12, 12))

for z, (u, v) in enumerate(frequencies):
    m, n = np.meshgrid(np.arange(N), np.arange(M))
    sine_wave = np.sin(2 * np.pi * (u * m / N + v * n / M))

    fft_2d = np.fft.fft2(sine_wave)
    fft_2d_magnitude = np.abs(np.fft.fftshift(fft_2d))

    plt.subplot(3, 2, z * 2 + 1)
    plt.imshow(sine_wave, cmap="gray", extent=[0, N, 0, M])
    plt.title(f"2D Sine Wave (u={u}, v={v})")
    plt.colorbar()

    plt.subplot(3, 2, z * 2 + 2)
    plt.imshow(fft_2d_magnitude, cmap="hot", extent=[-N // 2, N // 2, -M // 2, M // 2])
    plt.title(f"Fourier Transform (u={u}, v={v})")
    plt.colorbar()

plt.tight_layout()
plt.show()




#Q6: pyramid blending


def load_and_resize(image_path, size=(256, 256)):
    image = cv2.imread(image_path)
    return cv2.resize(image, size)

apple = load_and_resize(r"/Users/dwanith/Desktop/Semester 3/CV/HW6/homework6/data/apple.jpg")
orange = load_and_resize(r"/Users/dwanith/Desktop/Semester 3/CV/HW6/homework6/data/orange.jpg")



def generate_gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

apple_gp = generate_gaussian_pyramid(apple, 6)
orange_gp = generate_gaussian_pyramid(orange, 6)



def generate_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        next_level = gaussian_pyramid[i + 1]
        expected_size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        gaussian_expanded = cv2.pyrUp(next_level, dstsize=expected_size)
        laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Add the smallest level as it is
    return laplacian_pyramid

apple_lap_pyramid = generate_laplacian_pyramid(apple_gp)
orange_lap_pyramid = generate_laplacian_pyramid(orange_gp)




def plot_pyramid(pyramid, title):
    plt.figure(figsize=(10, 2))
    for i, layer in enumerate(pyramid):
        plt.subplot(1, len(pyramid), i + 1)
        plt.imshow(cv2.cvtColor(layer, cv2.COLOR_BGR2RGB))
        plt.title(f'Level {i}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()



plot_pyramid(apple_lap_pyramid, "Apple Laplacian Pyramid")
plot_pyramid(orange_lap_pyramid, "Orange Laplacian Pyramid")



#Extra credit


desert = cv2.imread(r"/Users/dwanith/Desktop/Semester 3/CV/HW6/homework6/data/desert.jpeg")
ocean = cv2.imread(r"/Users/dwanith/Desktop/Semester 3/CV/HW6/homework6/data/ocean.jpeg")

desert = cv2.resize(desert, (256, 256))
ocean = cv2.resize(ocean, (256, 256))

def generate_gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

desert_gp = generate_gaussian_pyramid(desert, 6)
ocean_gp = generate_gaussian_pyramid(ocean, 6)

def generate_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        next_level = gaussian_pyramid[i + 1]
        expanded = cv2.pyrUp(next_level, dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

desert_lp = generate_laplacian_pyramid(desert_gp)
ocean_lp = generate_laplacian_pyramid(ocean_gp)

def plot_pyramids(pyramids, titles):
    fig, axes = plt.subplots(nrows=len(pyramids), ncols=len(pyramids[0]), figsize=(20, 10))
    for ax_row, pyramid, title in zip(axes, pyramids, titles):
        for ax, layer in zip(ax_row, pyramid):
            ax.imshow(cv2.cvtColor(layer, cv2.COLOR_BGR2RGB))
            ax.axis('off')
        ax_row[0].set_ylabel(title)
    plt.show()

plot_pyramids([desert_lp, ocean_lp], ["Desert Laplacian Pyramid", "Ocean Laplacian Pyramid"])







