import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from numba import jit

@jit(nopython=True)
def sieve_of_eratosthenes(limit):
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    
    return np.where(is_prime)[0]

@jit(nopython=True)
def compute_transformations(primes):
    pct_change = np.empty(len(primes) - 1)
    for i in range(len(primes) - 1):
        pct_change[i] = (primes[i+1] - primes[i]) / primes[i]
    
    cumsum_pct = np.cumsum(pct_change)
    cos_cumsum = np.cos(cumsum_pct)
    sin_cumsum = np.sin(cumsum_pct)
    cumsum_cos = np.cumsum(cos_cumsum)
    cumsum_sin = np.cumsum(sin_cumsum)
    
    return pct_change, cumsum_pct, cos_cumsum, sin_cumsum, cumsum_cos, cumsum_sin

def compute_prime_cumsum_plot(limit):
    print(f"Generating primes up to {limit}...")
    primes = sieve_of_eratosthenes(limit)
    print(f"Found {len(primes)} primes. Computing transformations...")
    
    pct_change, cumsum_pct, cos_cumsum, sin_cumsum, cumsum_cos, cumsum_sin = compute_transformations(primes)

    print("Creating DataFrame and saving results...")
    
    # Use chunked writing for memory efficiency with large datasets
    chunk_size = 1000000
    output_csv = f"prime_cumsum_{limit}.csv"
    
    if len(primes) > chunk_size:
        # Write header
        pd.DataFrame(columns=["Prime", "Cumsum_Percentage_Change", "Cosine_of_Cumsum", "Sine_of_Cumsum", "Cumsum_of_Cosine", "Cumsum_of_Sine"]).to_csv(output_csv, index=False)
        
        # Write data in chunks
        for i in range(0, len(primes) - 1, chunk_size):
            end_idx = min(i + chunk_size, len(primes) - 1)
            chunk_df = pd.DataFrame({
                "Prime": primes[i:end_idx],
                "Cumsum_Percentage_Change": cumsum_pct[i:end_idx],
                "Cosine_of_Cumsum": cos_cumsum[i:end_idx],
                "Sine_of_Cumsum": sin_cumsum[i:end_idx],
                "Cumsum_of_Cosine": cumsum_cos[i:end_idx],
                "Cumsum_of_Sine": cumsum_sin[i:end_idx]
            })
            chunk_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        df = pd.DataFrame({
            "Prime": primes[:-1],
            "Cumsum_Percentage_Change": cumsum_pct,
            "Cosine_of_Cumsum": cos_cumsum,
            "Sine_of_Cumsum": sin_cumsum,
            "Cumsum_of_Cosine": cumsum_cos,
            "Cumsum_of_Sine": cumsum_sin
        })
        df.to_csv(output_csv, index=False)
    
    print(f"Saved CSV: {output_csv}")

    # Plot with downsampling for large datasets
    print("Creating plots...")
    plot_primes = primes[:-1]
    plot_cumsum_cos = cumsum_cos
    plot_cumsum_sin = cumsum_sin
    
    if len(plot_primes) > 100000:
        # Downsample for plotting
        step = len(plot_primes) // 50000
        plot_primes = plot_primes[::step]
        plot_cumsum_cos = plot_cumsum_cos[::step]
        plot_cumsum_sin = plot_cumsum_sin[::step]
        print(f"Downsampled to {len(plot_primes)} points for plotting")
    
    # Create subplots for both cumulative cosine, sine, and their relationship
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # Plot 1: Cumulative Sum of Cosine vs Prime Numbers
    ax1.plot(plot_primes, plot_cumsum_cos, linewidth=1, color='blue')
    ax1.set_title(f"Cumulative Sum of Cosine (Primes up to {limit})")
    ax1.set_xlabel("Prime Number")
    ax1.set_ylabel("Cumulative Sum of Cosine Values")
    ax1.grid(True)
    
    # Plot 2: Cumulative Sum of Sine vs Prime Numbers
    ax2.plot(plot_primes, plot_cumsum_sin, linewidth=1, color='red')
    ax2.set_title(f"Cumulative Sum of Sine (Primes up to {limit})")
    ax2.set_xlabel("Prime Number")
    ax2.set_ylabel("Cumulative Sum of Sine Values")
    ax2.grid(True)
    
    # Plot 3: Cumulative Sum of Sine vs Cumulative Sum of Cosine
    ax3.plot(plot_cumsum_cos, plot_cumsum_sin, linewidth=0.5, alpha=0.7, color='purple')
    ax3.set_title(f"Cumulative Sum of Sine vs Cumulative Sum of Cosine (Primes up to {limit})")
    ax3.set_xlabel("Cumulative Sum of Cosine Values")
    ax3.set_ylabel("Cumulative Sum of Sine Values")
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot 4: Both cumulative sums overlaid
    ax4.plot(plot_primes, plot_cumsum_cos, linewidth=1, color='blue', label='Cumsum Cosine', alpha=0.7)
    ax4.plot(plot_primes, plot_cumsum_sin, linewidth=1, color='red', label='Cumsum Sine', alpha=0.7)
    ax4.set_title(f"Both Cumulative Sums (Primes up to {limit})")
    ax4.set_xlabel("Prime Number")
    ax4.set_ylabel("Cumulative Sum Values")
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"prime_cumsum_combined_{limit}.png")
    print(f"Saved combined plot: prime_cumsum_combined_{limit}.png")
    
    # Also create individual high-resolution plots
    
    # Individual plot: Cumulative Sum of Sine vs Cumulative Sum of Cosine
    plt.figure(figsize=(12, 12), dpi=150)
    plt.plot(plot_cumsum_cos, plot_cumsum_sin, linewidth=0.5, alpha=0.7, color='purple')
    plt.title(f"Cumulative Sum of Sine vs Cumulative Sum of Cosine (Primes up to {limit})")
    plt.xlabel("Cumulative Sum of Cosine Values")
    plt.ylabel("Cumulative Sum of Sine Values")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"prime_sine_vs_cosine_cumsum_{limit}.png")
    print(f"Saved individual plot: prime_sine_vs_cosine_cumsum_{limit}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot cumulative sums of cosine and sine of cumulative percentage change of primes.")
    parser.add_argument("--limit", type=int, default=1000000, help="Upper limit for prime number generation")
    args = parser.parse_args()
    compute_prime_cumsum_plot(args.limit)