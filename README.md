
# SKA Prime Numbers: Real-Time Learning of the Prime Sequence

This project explores a radically new approach to understanding the prime numbers through the lens of **Structured Knowledge Accumulation (SKA)**—a forward-only, entropy-driven learning framework.

Unlike traditional number theory, which analyzes prime properties in a static or retrospective manner, this repository initiates a **real-time learning process** over the sequence of primes, with the goal of uncovering **hidden dynamics**, **regime transitions**, and **information structures** encoded in their progression.

## Clarifying "Real-Time Learning" in SKA

This project uses the term **“real-time learning”** not in the traditional sense of dynamic, time-stamped input, but to emphasize a **step-by-step, forward-only accumulation of information** over the prime number sequence.

In contrast to batch-style, retrospective analysis:

- **SKA processes primes one by one**, updating internal knowledge at each step,
- without revisiting earlier data or requiring access to the full sequence.

This mimics **temporal learning**, though “time” here refers to **symbolic sequence position** (index $n$) rather than physical time.

> **Key Insight:** This allows us to model prime numbers as an *informational stream*—letting structure emerge purely through accumulation.



##  Objectives

- Develop **real-time learning algorithms** to track evolving information in the prime number sequence.
- Define **novel features** on primes for SKA processing:

1. **Percentage gap changes:**

$$\Large \frac{P_{k+1} - P_k}{P_k}$$

2. **One-dimensional oscillatory projection:**

$$
f(n) = \sum_{k=1}^{n} \cos\left( \sum_{j=1}^{k} \frac{P_{j+1} - P_j}{P_j} \right)
$$

3. **Two-dimensional phase features:**

$$
     \Largex_k = \sum_{i=1}^{k} \cos\left( \sum_{j=1}^{i} \frac{P_{j+1} - P_j}{P_j} \right)
     \quad,\quad
      y_k = \sum_{i=1}^{k} \sin\left( \sum_{j=1}^{i} \frac{P_{j+1} - P_j}{P_j} \right)
$$

     These serve as the **feature trajectory** over which SKA will operate, offering a geometric embedding of prime information.

- Compare SKA learning trajectories over different feature representations.

- Examine **entropy evolution** and **alignment metrics** for uncovering latent order or geometric resonance in prime distributions.



## Why This Is Surprising

Prime gaps are famously irregular. However, by applying SKA and constructing transforms such as $f(n)$, $x_k$, and $y_k$, we begin to see **bounded, wave-like patterns** emerge.

This suggests that what appears to be randomness may conceal **hidden phase regularities**—challenging classical assumptions and opening new interdisciplinary pathways between number theory, signal processing, and information geometry.


##  Structure (Initial)




## Next Steps

* [x] Build and validate the prime-based feature transformations $\large f(n), x_k, y_k$
* [ ] Launch real-time SKA learning over the feature stream
* [ ] Add visualizations of entropy, alignment, and trajectory evolution
* [ ] Explore **phase portraits** and **Fourier projections** of $x_k, y_k$ trajectory
* [ ] Investigate deeper links between prime geometry and dynamical systems



