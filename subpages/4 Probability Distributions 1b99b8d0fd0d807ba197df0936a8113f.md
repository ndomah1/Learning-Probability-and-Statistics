# 4. Probability Distributions

# Binomial Distribution

The **Binomial Distribution** models the number of **successes** in n independent trials, each with a success probability p.

## Formula for Binomial Probability

$$
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
$$

Where:

- $n$ = total trials
- $k$ = number of successes
- $p$ = probability of success in each trial
- $\binom{n}{k}$ = **Combination formula:**
    
    $$
    \binom{n}{k} = \frac{n!}{k!(n-k)!}
    $$
    

## Expected Value and Variance

$$
\mathbb{E}[X] = np, \quad \mathrm{Var}(X) = np(1 - p)
$$

**Example:** If a fair coin is flipped **10 times**, what is the probability of getting exactly **6 heads**?

```python
from scipy.stats import binom

# Parameters
n_trials = 10
p_success = 0.5
k_successes = 6

# Compute probability P(X=6)
binomial_prob = binom.pmf(k_successes, n_trials, p_success)
binomial_prob
```

```
np.float64(0.2050781249999999)
```

The probability of getting exactly **6 heads** in **10 coin flips** is **0.205** (or **20.5%**).

# Poisson Distribution

The **Poisson Distribution** models the number of **events** in a fixed time interval, given an average rate $\lambda$.

## Formula for Poisson Distribution

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

## Expected Value and Variance

$$
\mathbb{E}[X] = \lambda, \quad \mathrm{Var}(X) = \lambda
$$

**Example:**

A website gets an average of **5 customer orders per hour**. What is the probability that exactly **7 orders** arrive in an hour?

```python
from scipy.stats import poisson

# Parameters
lambda_rate = 5 # Average rate (λ)
k_orders = 7 # Exact number of events

# Compute probability P(X=7)
poisson_prob = poisson.pmf(k_orders, lambda_rate)
poisson_prob
```

```
np.float64(0.10444486295705395)
```

The probability of receiving exactly **7 orders** in an hour is **0.104** (or **10.4%**).

# Normal Distribution

The **Normal (Gaussian) Distribution** is continuous and follows a **bell-shaped curve**, defined by:

- **Mean** ($\mu$): Center of the distribution.
- **Standard Deviation** ($\sigma$): Spread of the distribution.

## Probability Density Function (PDF)

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

**Example:**

A company’s employees have an average salary of **$50,000** with a **standard deviation of $8,000.** What is the probability that a randomly selected employee earns **less than $45,000**?

```python
# Normal distribution parameters
mu_salary = 50000 # Mean
sigma_salary = 8000 # Standard deviation
x_value = 45000 # Threshold value

# Compute probability P(X < 45000)
normal_prob = stats.norm.cdf(x_value, loc=mu_salary, scale=sigma_salary)
normal_prob
```

```
np.float64(0.26598552904870054)
```

The probability that a randomly selected employee earns **less than $45,000** is **0.266** (or **26.6%**).

# Exponential Distribution

The **Exponential Distribution** models the time between events in a Poisson process.

## Probability Density Function (PDF)

$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

## Expected Value and Variance

$$
\mathbb{E}[X] = \frac{1}{\lambda}, \quad \mathrm{Var}(X) = \frac{1}{\lambda^2}
$$

**Example:** 

A call center receives **2 calls per minute** ($\lambda = 2$). What is the probability that the next call arrives **after 1 minute**?

```python
# Exponential distribution parameters
lambda_rate = 2 # Calls per minute
x_time = 1 # Time threshold

# Compute probability P(X > 1) using survival function (1 - CDF)
exponential_prob = 1 - stats.expon.cdf(x_time, scale=1/lambda_rate)
exponential_prob
```

```
np.float64(0.1353352832366127)
```

The probability that the next call arrives **after 1 minute** is **0.135** (or **13.5%).**

# Understanding PDF vs. CDF

- **PDF (Probability Density Function):** Gives the **density** (not probability) at a specific X.
- **CDF (Cumulative Distribution Function):** Gives **P(X ≤ x).**

**Example:** Let’s visualize the **Normal Distribution’s PDF and CDF.**

```python
# Generate x values for plotting
x_values = np.linspace(mu_salary - 3*sigma_salary, mu_salary + 3*sigma_salary, 1000)

# Compute PDF and CDF
pdf_values = stats.norm.pdf(x_values, loc=mu_salary, scale=sigma_salary)
cdf_values = stats.norm.cdf(x_values, loc=mu_salary, scale=sigma_salary)

# Plot PDF and CDF
plt.figure(figsize=(12, 5))

# Plot PDF
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, label="PDF", color="blue")
plt.xlabel("Salary ($)")
plt.ylabel("Density")
plt.title("Normal Distribution PDF")
plt.legend()

# Plot CDF
plt.subplot(1, 2, 2)
plt.plot(x_values, cdf_values, label="CDF", color="green")
plt.xlabel("Salary ($)")
plt.ylabel("Cumulative Probability")
plt.title("Normal Distribution CDF")
plt.legend()
plt.tight_layout()

plt.show()
```

![image.png](4%20Probability%20Distributions%201b99b8d0fd0d807ba197df0936a8113f/image.png)

The plots above show:

- **PDF (left plot):** The bell curve that represents the density of salaries.
- **CDF (right plot):** The cumulative probability of salaries up to a given value.

This demonstrates how **PDF** helps find **densities,** while **CDF** helps calculating **probabilities.**

# Summary

| Distribution | Type | Key Parameter(s) | Example |
| --- | --- | --- | --- |
| **Binomial** | Discrete | $n, p$ | # of heads in 10 coin flips |
| **Poisson** | Discrete | $\lambda$ | # of customer arrivals per hour |
| **Normal** | Continuous | $\mu, \sigma^2$ | Height of people |
| **Exponential** | Continuous | $\lambda$ | Waiting time between events |