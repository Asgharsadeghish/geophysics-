# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:01:01 2024

@author: saber.sh
"""
#need-library
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
#
# Define the EMD function
def emd(signal, max_iterations=1000, stop_threshold=0.01):
    
    def is_monotonic(x):
        return np.all(np.diff(x) >= 0) or np.all(np.diff(x) <= 0)

    def compute_envelope(signal, extrema):
        t, values = extrema
        spline = CubicSpline(t, values)
        return spline(np.arange(len(signal)))

    # Initialize variables
    residual = signal.copy()
    imfs = []

    while not is_monotonic(residual):
        h = residual.copy()
        for _ in range(max_iterations):
            # Identify maxima and minima
            maxima = np.argwhere((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:])).flatten() + 1
            minima = np.argwhere((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:])).flatten() + 1

            # Boundary conditions for extrema
            if maxima.size == 0 or minima.size == 0:
                break
            maxima = np.insert(maxima, 0, 0) if maxima[0] != 0 else maxima
            minima = np.insert(minima, 0, 0) if minima[0] != 0 else minima
            maxima = np.append(maxima, len(h) - 1) if maxima[-1] != len(h) - 1 else maxima
            minima = np.append(minima, len(h) - 1) if minima[-1] != len(h) - 1 else minima

            # Calculate upper and lower envelopes
            upper_envelope = compute_envelope(h, (maxima, h[maxima]))
            lower_envelope = compute_envelope(h, (minima, h[minima]))

            # Compute the mean of the envelopes
            mean_envelope = (upper_envelope + lower_envelope) / 2

            # Update h
            prev_h = h.copy()
            h -= mean_envelope

            #  stop inner loop
            if np.linalg.norm(h - prev_h) < stop_threshold:
                break

        # Save_IMF
        if maxima.size > 0 and minima.size > 0:
            imfs.append(h)

        # Update residual
        residual -= h

    # Append residual as the last IMF
    imfs.append(residual)

    return imfs

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate a sample signal (e.g., a combination of sinusoids)
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    signal = clean_signal + 0.2 * np.random.randn(len(t))

    # Perform EMD
    imfs = emd(signal)

    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, len(imfs) * 2))

    plt.subplot(len(imfs) + 1, 1, 1)
    plt.plot(t, clean_signal, label="clean_signal", color="red")
    plt.title("Original Signal ")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    
    
    plt.figure(figsize=(12, len(imfs) * 2))

    plt.subplot(len(imfs) + 1, 1, 1)
    plt.plot(t, signal, label="Original Signal (with noise)", color="blue")
    plt.title("Original Signal (with noise)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()

    for i, imf in enumerate(imfs):
        plt.subplot(len(imfs) + 1, 1, i + 2)
        plt.plot(t, imf, label=f"IMF {i + 1}", color="green")
        plt.title(f"IMF {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()

    # Combine IMFs from the four to the second last
    denoised_signal = np.sum(imfs[4:-1], axis=0)

    # Plot clean signal vs. denoised signal
    plt.figure(figsize=(12, 6))
    plt.plot(t, clean_signal, label="Original Signal (without noise)", color="blue", linestyle="--")
    plt.plot(t, denoised_signal, label=" (Sum of IMFs 4 to N-1)", color="red")
    plt.title("Comparison of Original Signal (without noise) and Denoised Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


