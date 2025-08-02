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

@jit(nopython=True)
def find_intersections(cumsum_cos, cumsum_sin, tolerance=1e-6):
    intersections = []
    
    for i in range(len(cumsum_cos)):
        diff = abs(cumsum_cos[i] - cumsum_sin[i])
        if diff <= tolerance:
            intersections.append(i)
    
    return np.array(intersections)

@jit(nopython=True)
def find_sign_change_intersections(cumsum_cos, cumsum_sin):
    intersections = []
    
    for i in range(1, len(cumsum_cos)):
        diff_prev = cumsum_cos[i-1] - cumsum_sin[i-1]
        diff_curr = cumsum_cos[i] - cumsum_sin[i]
        
        # Check if there's a sign change (intersection)
        if diff_prev * diff_curr <= 0:
            # Linear interpolation to find more precise intersection point
            if abs(diff_curr - diff_prev) > 1e-12:
                t = -diff_prev / (diff_curr - diff_prev)
                intersections.append(i - 1 + t)
            else:
                intersections.append(i)
    
    return np.array(intersections)

def find_prime_intersections(limit, tolerance=1e-6, use_interpolation=True):
    print(f"Generating primes up to {limit}...")
    primes = sieve_of_eratosthenes(limit)
    print(f"Found {len(primes)} primes. Computing transformations...")
    
    pct_change, cumsum_pct, cos_cumsum, sin_cumsum, cumsum_cos, cumsum_sin = compute_transformations(primes)
    
    print("Finding intersections where cumsum_cosine ≈ cumsum_sine...")
    
    if use_interpolation:
        intersection_indices = find_sign_change_intersections(cumsum_cos, cumsum_sin)
        print(f"Found {len(intersection_indices)} intersections using sign change detection")
    else:
        intersection_indices = find_intersections(cumsum_cos, cumsum_sin, tolerance)
        print(f"Found {len(intersection_indices)} intersections within tolerance {tolerance}")
    
    # Extract intersection data
    intersection_data = []
    
    for idx in intersection_indices:
        if use_interpolation and not idx.is_integer():
            # Interpolated intersection
            i = int(idx)
            frac = idx - i
            
            if i + 1 < len(primes) - 1:
                # Interpolate values
                prime_interp = primes[i] + frac * (primes[i+1] - primes[i])
                cumsum_cos_interp = cumsum_cos[i] + frac * (cumsum_cos[i+1] - cumsum_cos[i])
                cumsum_sin_interp = cumsum_sin[i] + frac * (cumsum_sin[i+1] - cumsum_sin[i])
                
                intersection_data.append({
                    'Index': idx,
                    'Prime_Interpolated': prime_interp,
                    'Cumsum_Cosine': cumsum_cos_interp,
                    'Cumsum_Sine': cumsum_sin_interp,
                    'Difference': abs(cumsum_cos_interp - cumsum_sin_interp)
                })
        else:
            # Exact index intersection
            i = int(idx)
            if i < len(primes) - 1:
                intersection_data.append({
                    'Index': i,
                    'Prime_Number': primes[i],
                    'Cumsum_Cosine': cumsum_cos[i],
                    'Cumsum_Sine': cumsum_sin[i],
                    'Difference': abs(cumsum_cos[i] - cumsum_sin[i])
                })
    
    # Create DataFrame
    if intersection_data:
        df_intersections = pd.DataFrame(intersection_data)
        output_csv = f"prime_intersections_{limit}.csv"
        df_intersections.to_csv(output_csv, index=False)
        print(f"Saved {len(intersection_data)} intersections to: {output_csv}")
        
        # Display first few intersections
        print("\nFirst 10 intersections:")
        print(df_intersections.head(10).to_string(index=False))
        
        if len(intersection_data) > 10:
            print(f"\n... and {len(intersection_data) - 10} more intersections")
    else:
        print("No intersections found!")
        return
    
    # Create visualization
    print("Creating visualization...")
    
    # Downsample for plotting if needed
    plot_primes = primes[:-1]
    plot_cumsum_cos = cumsum_cos
    plot_cumsum_sin = cumsum_sin
    plot_intersections = intersection_data
    
    if len(plot_primes) > 100000:
        step = len(plot_primes) // 50000
        plot_primes = plot_primes[::step]
        plot_cumsum_cos = plot_cumsum_cos[::step]
        plot_cumsum_sin = plot_cumsum_sin[::step]
        print(f"Downsampled to {len(plot_primes)} points for plotting")
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # Plot 1: Both cumulative sums with intersection points
    ax1.plot(plot_primes, plot_cumsum_cos, linewidth=1, color='blue', label='Cumsum Cosine', alpha=0.7)
    ax1.plot(plot_primes, plot_cumsum_sin, linewidth=1, color='red', label='Cumsum Sine', alpha=0.7)
    
    # Mark intersection points
    if use_interpolation:
        intersection_primes = [d.get('Prime_Interpolated', d.get('Prime_Number', 0)) for d in plot_intersections[:50]]  # Limit to first 50 for visibility
        intersection_values = [d['Cumsum_Cosine'] for d in plot_intersections[:50]]
    else:
        intersection_primes = [d['Prime_Number'] for d in plot_intersections[:50]]
        intersection_values = [d['Cumsum_Cosine'] for d in plot_intersections[:50]]
    
    ax1.scatter(intersection_primes, intersection_values, color='green', s=20, zorder=5, label=f'Intersections ({len(intersection_data)} total)')
    ax1.set_title(f"Cumulative Sums with Intersections (Primes up to {limit})")
    ax1.set_xlabel("Prime Number")
    ax1.set_ylabel("Cumulative Sum Values")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Difference between cumulative sums
    diff = plot_cumsum_cos - plot_cumsum_sin
    ax2.plot(plot_primes, diff, linewidth=1, color='purple')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title("Difference: Cumsum_Cosine - Cumsum_Sine")
    ax2.set_xlabel("Prime Number")
    ax2.set_ylabel("Difference")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parametric plot with intersection points
    ax3.plot(plot_cumsum_cos, plot_cumsum_sin, linewidth=0.5, alpha=0.7, color='purple')
    ax3.plot([-1000, 1000], [-1000, 1000], 'k--', alpha=0.5, label='y = x line')
    
    if intersection_data:
        intersection_cos = [d['Cumsum_Cosine'] for d in plot_intersections[:100]]
        intersection_sin = [d['Cumsum_Sine'] for d in plot_intersections[:100]]
        ax3.scatter(intersection_cos, intersection_sin, color='green', s=20, zorder=5, label='Intersections')
    
    ax3.set_title("Parametric Plot: Cumsum_Sine vs Cumsum_Cosine")
    ax3.set_xlabel("Cumulative Sum of Cosine")
    ax3.set_ylabel("Cumulative Sum of Sine")
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    ax3.legend()
    
    # Plot 4: Histogram of intersection differences
    if intersection_data:
        differences = [d['Difference'] for d in intersection_data]
        ax4.hist(differences, bins=50, alpha=0.7, color='orange')
        ax4.set_title("Distribution of Intersection Differences")
        ax4.set_xlabel("Absolute Difference")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"prime_intersections_analysis_{limit}.png")
    print(f"Saved analysis plot: prime_intersections_analysis_{limit}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find intersections where cumsum_cosine equals cumsum_sine for prime analysis.")
    parser.add_argument("--limit", type=int, default=1000000, help="Upper limit for prime number generation")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for intersection detection")
    parser.add_argument("--no-interpolation", action="store_true", help="Disable interpolation-based intersection finding")
    args = parser.parse_args()
    
    find_prime_intersections(args.limit, args.tolerance, not args.no_interpolation)