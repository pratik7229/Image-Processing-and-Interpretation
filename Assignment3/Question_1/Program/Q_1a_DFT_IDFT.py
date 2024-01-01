import numpy as np
import matplotlib.pyplot as plt

# Define the signal
f = np.array([2, 3, 4, 4], dtype=complex)

# Number of samples
N = len(f)

# Compute the DFT of a singnal
def DFT(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for u in range(N):
        X[u] = sum(signal[n] * np.exp(-2j * np.pi * u * n / N) for n in range(N))/N
    return X

dft = DFT(f)

# Compute the inverse DFT of a signal
def inverse_DFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        x[n] = sum(X[u] * np.exp(2j * np.pi * u * n / N) for u in range(N))
    return x

inverse_dft = inverse_DFT(dft)

# Calculate and plot the real and imaginary parts of the DFT
real_part_DFT = [np.real(coeff) for coeff in dft]
imaginary_part_DFT = [np.imag(coeff) for coeff in dft]

# Calculate and plot the magnitude of the DFT
magnitude_DFT = [np.sqrt(np.real(coeff) ** 2 + np.imag(coeff) ** 2) for coeff in dft]

# Calculate and plot the real and imaginary parts of the Inverse DFT
real_part_IDFT = [np.real(coeff) for coeff in inverse_dft]
imaginary_part_IDFT = [np.imag(coeff) for coeff in inverse_dft]

# Calculate and plot the magnitude of the Inverse DFT
magnitude_IDFT = [np.sqrt(np.real(coeff) ** 2 + np.imag(coeff) ** 2) for coeff in inverse_dft]

# Plot the Input signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.stem(f)
plt.title('Input signal')

# Plot the real part of the DFT
plt.subplot(2, 3, 2)
plt.stem(real_part_DFT)
plt.title('Real Part of DFT')

# Plot the imaginary part of the DFT
plt.subplot(2, 3, 3)
plt.stem(imaginary_part_DFT)
plt.title('Imaginary Part of DFT')

# Plot the magnitude of the DFT
plt.subplot(2, 3, 4)
plt.stem(magnitude_DFT)
plt.title('Magnitude of DFT')

# Plot the Inverse of the DFT
plt.subplot(2, 3, 5)
plt.stem(inverse_dft)
plt.title('Inverse of DFT')

# Show the plots
plt.tight_layout()
plt.show()
