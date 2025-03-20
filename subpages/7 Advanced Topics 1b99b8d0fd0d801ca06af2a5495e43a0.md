# 7. Advanced Topics

# Time Series Analysis

Time series analysis is essential when dealing with data collected over time (e.g, sales, stock prices, temperatures).

## Key Concepts

- **Trend:** A long-term increase or decrease in data.
- **Seasonality:** Recurring patterns over a fixed time (e.g., yearly sales fluctuations).
- **Autocorrelation:** The correlation of a time series with a lagged version of itself.
- **Stationarity:** A time series is stationary if its statistical properties (mean, variance) remain constant over time.

## Time Series Forecasting with ARIMA

Let’s analyze a time series dataset and apply an **ARIMA model** for forecasting.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Simulate a time series (e.g., monthly sales data)
np.random.seed(42)
time_series = pd.Series(
    np.cumsum(np.random.randn(100) + 0.5),
    index=pd.date_range(start="2020-01-01", periods=100, freq='M')
)

# Check for stationarity using the Augmented Dickey-Fuller test
adf_test = adfuller(time_series)
adf_result = {"Test Statistic": adf_test[0], "p-value": adf_test[1]}
print("ADF test result:", adf_result)

# Fit ARIMA model (Auto-Regressive Integrated Moving Average)
model = ARIMA(time_series, order=(2, 1, 2))  # ARIMA(p,d,q) with differencing (d=1)
model_fit = model.fit()

# Forecast the next 12 months
forecast = model_fit.forecast(steps=12)

# Plot original data and forecast
plt.figure(figsize=(10, 5))
plt.plot(time_series, label="Observed Time Series", color="blue")
plt.plot(forecast.index, forecast, label="ARIMA Forecast", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Forecasting using ARIMA")
plt.legend()
plt.show()
```

![image.png](7%20Advanced%20Topics%201b99b8d0fd0d801ca06af2a5495e43a0/image.png)

The **ARIMA model** was fitted, and the **forecasted values** for the next 12 months are shown in red.

- **ADF Test Results:**
    - **p-value = 0.97**, indicating that the time series is **not stationary** (suggesting we may need further differencing or transformations).
    - This means we need to **remove trends** before applying time series models effectively.

# Bayesian Statistics

Bayesian statistics incorporates prior knowledge to update beliefs using **Bayes’ Theorem:**

$$
P(A | B) = \frac{P(B | A) P(A)}{P(B)}
$$

## Example: Bayesian Updating for Coin Bias

Let’s update our belief about a coin’s fairness after flipping it multiple times.

```python
import pymc3 as pm

# Prior belief: Coin is fair (P(Heads) = 0.5)
prior_heads = 0.5

# Observed Data: 10 flips, 7 were heads
n_flips = 10
n_heads = 7

# Bayesian model to estimate coin bias
with pm.Model():
 p = pm.Beta("p", alpha=1, beta=1) # Prior: Uniform (Beta(1,1))
 obs = pm.Binomial("obs", n=n_flips, p=p, observed=n_heads)
 trace = pm.sample(1000, return_inferencedata=False)
 
# Plot posterior distribution of coin bias
plt.figure(figsize=(8, 4))
sns.histplot(trace["p"], bins=30, kde=True)
plt.xlabel("Probability of Heads")
plt.ylabel("Density")
plt.title("Posterior Distribution of Coin Bias (Bayesian Inference)")
plt.show()
```

![image.png](7%20Advanced%20Topics%201b99b8d0fd0d801ca06af2a5495e43a0/image%201.png)

Key Points:

- **Peak Around ~0.7**: The distribution is centered near 0.7, indicating that—given these 7 heads out of 10 flips—the most likely estimate for the coin’s bias is around 70% heads.
- **Spread Reflects Uncertainty**: The posterior is spread out across a range of values from roughly 0.4 to 0.9. This spread shows the uncertainty in the estimate when you only have 10 flips of data.
- **Bayesian Interpretation**: Instead of giving a single number, the Bayesian approach provides a full distribution of plausible values for the coin’s bias, reflecting both the prior assumption (here, uniform) and the observed data.

# Resampling Techniques (Bootstrapping & Monte Carlo Simulations)

## Bootstrapping

Bootstrapping resamples from the observed data **with replacement** to estimate confidence intervals.

### Example: Bootstrapping Mean Salary

Let’s compute a **95% confidence interval** for the **mean salary** using bootstrapping.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample salary data
salary_data = np.array([50000, 60000, 75000, 82000, 95000, 105000, 115000])

# Bootstrap resampling (1000 samples)
n_iterations = 1000
bootstrap_means = [
    np.mean(np.random.choice(salary_data, size=len(salary_data), replace=True))
    for _ in range(n_iterations)
]

# Compute 95% confidence interval
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

# Plot bootstrap mean distribution
plt.figure(figsize=(8, 4))
sns.histplot(bootstrap_means, bins=30, kde=True, color="purple")
plt.axvline(ci_lower, color="red", linestyle="dashed", label="95% CI Lower")
plt.axvline(ci_upper, color="red", linestyle="dashed", label="95% CI Upper")
plt.xlabel("Bootstrap Mean Salary")
plt.ylabel("Frequency")
plt.title("Bootstrapped Confidence Interval for Mean Salary")
plt.legend()
plt.show()
```

![image.png](7%20Advanced%20Topics%201b99b8d0fd0d801ca06af2a5495e43a0/image%202.png)

The **95% Confidence Interval** for the mean salary is (**$67,143**, **$99,857**).

- This method allows us to **estimate uncertainty** in the mean without relying on strict normality assumptions.
- The **bootstrap histogram** shows the **distribution of sample means** from resampling.

## Monte Carlo Simulation

Monte Carlo simulations **simulate random experiments** to approximate probabilities.

### Example: Estimating $\pi$ Using Monte Carlo

We randomly sample points inside a square and count how many fall inside a circle.

```python
# Monte Carlo Simulation to estimate π
n_simulations = 10000
inside_circle = 0

# Generate random points (x, y)
x_random = np.random.uniform(-1, 1, n_simulations)
y_random = np.random.uniform(-1, 1, n_simulations)

# Check if points fall inside the unit circle (x^2 + y^2 <= 1)
inside_circle = np.sum(x_random**2 + y_random**2 <= 1)

# Estimate π
pi_estimate = (inside_circle / n_simulations) * 4

# Plot sampled points
plt.figure(figsize=(6, 6))
plt.scatter(x_random, y_random, s=1, alpha=0.5, label="Points")
circle = plt.Circle((0, 0), 1, color="red", fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Monte Carlo Estimation of π: {pi_estimate:.4f}")
plt.legend(["Unit Circle", "Random Points"])
plt.show()

# Display estimated π
pi_estimate
```

![image.png](7%20Advanced%20Topics%201b99b8d0fd0d801ca06af2a5495e43a0/image%203.png)

```
np.float64(3.166)
```

The **Monte Carlo simulation** estimated **$\pi \approx 3.166$**, which is close to the actual value (**3.1416**).

- **How it works:** We **randomly sample points** inside a square, then count how many land inside a **unit circle.**
- **Increasing simulations** improves accuracy.

# Summary

| Topic | Description |
| --- | --- |
| **Time Series Analysis** | Forecasting using ARIMA, handling trends & seasonality |
| **Bayesian Statistics** | Updating beliefs using prior distributions |
| **Bootstrapping** | Estimating confidence intervals by resampling data |
| **Monte Carlo Simulation** | Estimating probabilities via random sampling |