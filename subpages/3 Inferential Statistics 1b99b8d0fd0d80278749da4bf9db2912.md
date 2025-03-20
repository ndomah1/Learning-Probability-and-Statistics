# 3. Inferential Statistics

# Sampling Techniques and Central Limit Theorem (CLT)

## Sampling Techniques

### Simple Random Sampling

Each member of the population has an equal chance of being selected.

### Stratified Sampling

The population is divided into strata (subgroups), and random samples are taken from each.

### Systematic Sampling

Every $k$-th member is chosen from a list.

### Cluster Sampling

The population is divided into clusters, and a few clusters are randomly selected.

### Convenience Sampling

The sample is chosen based on ease of access (not always representative).

## Central Limit Theorem (CLT)

The **CLT** states that as the sample size $n$ increases, the **distribution of the sample mean** approximates a normal distributions, **regardless of the original data distribution.**

## Example: CLT Simulation

Let’s take a **skewed population distribution** and compute the **sample means** for increasing $n$ to see if they approach normality.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a skewed population (Exponential Distribution)
np.random.seed(42)
population = np.random.exponential(scale=2, size=10000)

# Function to draw sample means for different sample sizes
sample_sizes = [5, 30, 100]
sample_means = {n: [np.mean(np.random.choice(population, size=n)) for _ in range(1000)] for n in sample_sizes}

# Plot histograms of sample means
plt.figure(figsize=(15, 5))
for i, n in enumerate(sample_sizes, 1):
		plt.subplot(1, 3, i)
		sns.histplot(sample_means[n], kde=True, bins=30)
		plt.title(f"Sample Size {n} (CLT)")
		plt.xlabel("Sample Mean")
		plt.ylabel("Frequence")
		
plt.tight_layout()
plt.show()
```

![image.png](3%20Inferential%20Statistics%201b99b8d0fd0d80278749da4bf9db2912/image.png)

# Hypothesis Testing

## Key Concepts

1. **Null Hypothesis** ($H_0$) - Assumes no effect or difference.
2. **Alternative Hypothesis** ($H_1$) - What we suspect might be true.
3. **p-value** - The probability of observing results as extreme as the sample under $H_0$.
4. **Significance Level** ($\alpha$) - Commonly **0.05** (5% chance of false positives).
5. **Decision Rule**:
    1. If $p < \alpha$, **reject** $H_0$ (statistically significant).
    2. If $p > \alpha$, **fail to reject** $H_0$ (no strong evidence against it.

## Example: Hypothesis Test on Coin Fairness

A coin is flipped 100 times, and **90 heads** are observed. Is the coin biased?

- $H_0$: The coin is fair ($p = 0.5$).
- $H_1$: The coin is biased ($p ≠ 0.5$).
- **Test:** Binomial test for deviation from 50% heads.

Let’s compute the **p-value.**

```python
from scipy.stats import binomtest

# Hypothesis test for coin fairness (90 heads in 100 flips)
n_flips = 100
observed_heads = 90
expected_prob = 0.5

# Perform Binomial Test using binomtest
result = binomtest(observed_heads, n_flips, expected_prob, alternative='two-sided')
p_value = result.pvalue

print(p_value)
```

```
3.0632901754379845e-17
```

The **p-value** for the hypothesis test is $3.06\times10^{-17}$, which is **extremely small** ($< 0.05$).

Since $p < 0.05$, we **reject** $H_0$ and conclude that **the coin is likely biased.**

# Confidence Intervals

A **Confidence Interval** (**CI**) estimates the range where a population parameter (e.g., mean) is likely to be.

$$
\text{CI} = \bar{x} \pm Z \cdot \frac{\sigma}{\sqrt{n}}
$$

Where:

- $\bar{x}$ = Sample mean
- $Z$ = Critical value (1.96 for 95& CI)
- $\sigma$ = Standard deviation
- $n$ = Sample Size

## Example: Confidence Interval for Population Mean

Let’s compute a **95% confidence interval** for the mean of a sample.

```python
import scipy.stats as stats

# Given sample data
sample_data = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)  # Sample standard deviation
n = len(sample_data)

# Compute 95% Confidence Interval
confidence = 0.95
z_score = stats.t.ppf((1 + confidence) / 2, df=n-1)  # Using t-distribution for small sample size
margin_of_error = z_score * (sample_std / np.sqrt(n))
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

(ci_lower, ci_upper)
```

```
(np.float64(61.67074705139159), np.float64(83.32925294860841))
```

The **95% Confidence Interval (CI)** for the population mean is (**61.67**, **83.33**).

This means we are **95% confident** that the true mean lies within this range.

# t-tests, Chi-square Tests, and ANOVA

## t-test (Comparing Means)

- A **t-test** determines if two sample means are significantly different.
- **Types:**
    - One-sample t-test: Tests if the sample mean differs from a known value.
    - Two-sample t-test: Compares means of **two independent groups.**
    - Paired t-test: Compares **before/after** changes for the same group.

### Example: Two-Sample t-test

Let’s compare the exam scores of **two student groups.**

```python
# Sample exam scores for two groups
group_A = np.array([78, 82, 85, 88, 90, 92, 95, 97, 99, 100])
group_B = np.array([72, 75, 78, 80, 83, 85, 87, 89, 90, 92])

# Perform independent two-sample t-test
t_stat, p_value_ttest = stats.ttest_ind(group_A, group_B, equal_var=False)

# Display results
(t_stat, p_value_ttest)
```

```
(np.float64(2.3752738320732614), np.float64(0.028967453915632595))
```

The **t-test results** are:

- **t-statistics** = 2.375
- **p-value** = 0.029

Since $p < 0.05$, we reject $H_0$ and conclude that the two groups have significantly different mean exam scores.

## Chi-square Test (Categorical Data)

Used to test independence between **categorical variables** (e.g., gender vs preference).

### Example: Chi-square Test on Survey Responses

We test if **survey responses** (**Yes**/**No**) differ across two groups.

```python
from scipy.stats import chi2_contingency

# Contingency Table (Survey responses: Yes/No for two groups)
observed = np.array([[30, 10],   # Group A (Yes: 30, No: 10)
										 [20, 20]])  # Group B (Yes: 20, No: 20)
										 
# Perform Chi-square Test
chi2_stat, p_value_chi2, dof, expected = chi2_contingency(observed)

# Display results
(chi2_stat, p_value_chi2)
```

```
(np.float64(4.32), np.float64(0.03766692222862869))
```

The **Chi-square test results** are:

- **Chi-square statistics** = 4.32
- **p-value** = **0.037**

Since $p < 0.05$, we **reject** $H_0$ and conclude that the survey responses **differ significantly between the two groups.**

## ANOVA (Comparing More Than Two Groups)

- **ANOVA** (**Analysis of Variance**) checks if **3 or more group means** differ significantly.
- Uses an **F-statistic** (ratio of variance between groups to variance within groups).

### Example: One-Way ANOVA on Student Scores

We compare **test scores** from **three teaching methods.**

```python
# Sample test scores for three teaching methods
method_1 = np.array([85, 87, 90, 92, 95])
method_2 = np.array([78, 80, 82, 85, 88])
method_3 = np.array([70, 72, 75, 78, 80])

# Perform One-Way ANOVA
f_stat, p_value_anova = stats.f_oneway(method_1, method_2, method_3)

# Display results
(f_stat, p_value_anova)
```

```
(np.float64(16.94020618556702), np.float64(0.00032012795496790783))
```

The **ANOVA test results** are:

- **F-statistic** = 16.94
- **p-value** = **0.00032**

Since $p < 0.05$, we **reject** $H_0$ and conclude that **at least one teaching method leads to significantly different scores.**

# Correlation and Regression Analysis

## Correlation

### Pearson Correlation Coefficient

$$
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}
$$

### Example: Correlation Between Study Hours and Exam Scores

```python
# Sample Data: Study hours vs Exam scores
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# Compute Pearson correlation
correlation_coefficient, _ = stats.pearsonr(study_hours, exam_scores)

# Display correlation
correlation_coefficient
```

```
np.float64(0.9999999999999997)
```

The **Pearson correlation coefficient** ($r$) is **1.0**, indicating a **perfect positive correlation** between study hours and exam scores (as study hours increase, exam scores increase proportionally).

## Linear Regression

### Equation of Regression Line

$$
Y = \alpha + \beta X + \epsilon
$$

- $\beta$ (slope): Effect of $X$ on $Y$.
- **Intercept** ($\alpha$): Value of $Y$ when $X = 0$.

### Example: Predicting Exam Scores from Study Hours

```python
from sklearn.linear_model import LinearRegression

# Reshape study hours for sklearn
X = study_hours.reshape(-1, 1)
y = exam_scores

# Fit Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Regression Equation
(slope, intercept)
```

```
(np.float64(5.000000000000002), np.float64(44.999999999999986))
```

The **linear regression equation** is:

$$
\hat{Y} = 5X + 45
$$

This means that **each additional study hour increases the predicted exam score by 5 points**, starting from **45 points at 0 hours**.