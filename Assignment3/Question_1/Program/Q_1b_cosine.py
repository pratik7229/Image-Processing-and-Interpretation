import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 128
u = 8
x = np.arange(N)
cosine_wave = np.cos(2 * np.pi * u * x / N)

# Plot the original cosine wave
plt.figure(figsize=(12, 4))
plt.subplot(2, 3, 1)
plt.plot(x, cosine_wave, label=r'$\cos(2 \pi u x / N)$')
plt.title('Original Cosine Wave')


# Compute the DFT manually
def DFT(signal):
    N = len(signal)
    result = np.zeros(N, dtype=np.complex128)
    
    for k in range(N):
        for n in range(N):
            result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    
    return result

dft_result = DFT(cosine_wave)

# Plot the DFT components with shifted magnitude
shift_amount = N // 2

# Real Part with circular shift
real_part = np.empty_like(dft_result.real)
real_part[:shift_amount] = dft_result.real[-shift_amount:]
real_part[shift_amount:] = dft_result.real[:-shift_amount]

# Plot the DFT components
plt.subplot(2, 3, 2)
plt.plot(real_part, label='Real Part')
plt.title('Real Part of DFT')

# Imaginary Part with circular shift
imaginary_part = np.empty_like(dft_result.imag)
imaginary_part[:shift_amount] = dft_result.imag[-shift_amount:]
imaginary_part[shift_amount:] = dft_result.imag[:-shift_amount]

plt.subplot(2, 3, 3)
plt.plot(imaginary_part, label='Imaginary Part')
plt.title('Imaginary Part of DFT')

# Compute the magnitude and phase
magnitude = np.sqrt(dft_result.real**2 + dft_result.imag**2)
phase = np.arctan2(dft_result.imag, dft_result.real)

# Magnitude with circular shift
magnitude_shifted = np.empty_like(magnitude)
magnitude_shifted[:shift_amount] = magnitude[-shift_amount:]
magnitude_shifted[shift_amount:] = magnitude[:-shift_amount]

plt.subplot(2, 3, 4)
plt.plot(magnitude_shifted, label='Magnitude')
plt.title('Magnitude of Cosine wave')

# Phase with circular shift
phase_shifted = np.empty_like(phase)
phase_shifted[:shift_amount] = phase[-shift_amount:]
phase_shifted[shift_amount:] = phase[:-shift_amount]

plt.subplot(2, 3, 5)
plt.plot(phase_shifted, label='Phase')
plt.title('Phase part')

plt.tight_layout()
plt.show()
