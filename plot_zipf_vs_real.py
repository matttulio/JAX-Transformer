import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

# Set plot parameters
plt.style.use('science')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['figure.figsize'] = [11, 8]

save_dir = "Empirics/zipf/"
os.makedirs(save_dir, exist_ok=True)

colors = plt.cm.Paired

# Load frequency data
freqs = np.loadtxt('Datasets/real_freqs.txt')
freqs = freqs / np.sum(freqs)
freqs = np.sort(freqs)[::-1]

# Plot the frequencies
plt.plot(range(1, len(freqs) + 1), freqs, '-', linewidth=2, label='WikiText dataset', color=colors(9))

# Generate Zipf's law curve
ranks = np.arange(1, len(freqs) + 1)
zipf = 1 / ranks
zipf = zipf / np.sum(zipf)

# Plot Zipf's law for comparison
plt.plot(ranks, zipf, '--', color=colors(7), linewidth=2, label="Zipf's Law")

# Zipf-Mandelbrot law: f(k) ∝ 1 / (k + q)^s
q = 2.7 
s = 1.2
zipf_mandelbrot = 1 / (ranks + q) ** s
zipf_mandelbrot = zipf_mandelbrot / np.sum(zipf_mandelbrot)

# Plot Zipf-Mandelbrot law for comparison
plt.plot(ranks, zipf_mandelbrot, '-.', color=colors(5), linewidth=2, label="Zipf-Mandelbrot Law")


# Plot formatting
plt.xlabel('Rank', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14)
plt.title("Token Frequency Distribution vs Zipf's Law", fontsize=16)
plt.savefig(os.path.join(save_dir, "zipf_vs_real.png"), format='png', dpi=200)
