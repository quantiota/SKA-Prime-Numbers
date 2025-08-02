
# SKA Prime Numbers: Real-Time Learning of the Prime Sequence

This project explores a radically new approach to understanding the prime numbers through the lens of **Structured Knowledge Accumulation (SKA)**—a forward-only, entropy-driven learning framework.

Unlike traditional number theory, which analyzes prime properties in a static or retrospective manner, this repository initiates a **real-time learning process** over the sequence of primes, with the goal of uncovering **hidden dynamics**, **regime transitions**, and **information structures** encoded in their progression.

##  Objectives

- Develop **real-time learning algorithms** to track evolving information in the prime number sequence.
- Define **novel features** on primes for SKA processing:

  1. **Percentage gap changes:**

    $$
    \Large \frac{P_{k+1} - P_k}{P_k}
    $$

  2. **Oscillatory function:**

$$
\Large f(n) = \sum_{k=1}^{n} \cos\left( \sum_{j=1}^{k} \frac{P_{j+1} - P_j}{P_j} \right)
$$

     capturing cumulative angular distortion of prime gaps.

- Compare SKA learning trajectories over different feature representations.
- Examine **entropy evolution** and **alignment metrics** for uncovering latent order or geometric resonance in prime distributions.

##  Why this is surprising

Prime gaps are known to behave irregularly. However, through transformations like $\large f(n)$, we begin to see **bounded wave-like structures** emerge—suggesting that what appears to be randomness may in fact conceal **hidden phase regularities**.

This challenges classical assumptions and opens new interdisciplinary possibilities, linking number theory, signal processing, and information geometry.

## Structure (initial)



## Next Steps

- Compute the features
- Add real-time visualizations for entropy and cosine evolution
- Explore phase portraits and Fourier transforms




