# 5. Regression and Correlation Analysis

# Simple and Multiple Linear Regression

Linear regression models the relationship between one or more independent variables ($X)$ and a dependent variable ($Y$).

## Simple Linear Regression

### Equation

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

Where:

- $Y$ = dependent variable (target)
- $X$ = independent variable (predictor)
- $\beta_0$ = intercept
- $\beta_1$ = slope (effect of $X$ on $Y$)
- $\epsilon$ = error term (residuals)

### Example: Predicting House Prices Based on Size

Let’s fit a simple linear regression model to predict **house prices** based on **size (sq ft).**

```python
# Sample data: House Size (sq ft) vs Price ($1000s)
house_size = np.array([850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300])
house_price = np.array([200, 210, 220, 230, 240, 250, 260, 270, 280, 290])

# Reshape for sklearn
X = house_size.reshape(-1, 1)
y = house_price

# Fit Simple Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Get regression coefficients
slope_simple = model.coef_[0] # β1
intercept_simple = model.intercept_ # β0

# Predict values
y_pred = model.predict(X)

# Plot regression line
plt.figure(figsize=(8, 5))
plt.scatter(house_size, house_price, color="blue", label="Actual Data")
plt.plot(house_size, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($1000s)")
plt.title("Simple Linear Regression: House Size vs Price")
plt.legend()
plt.show()

# Display regression equation
(slope_simple, intercept_simple)
```

![image.png](5%20Regression%20and%20Correlation%20Analysis%201b99b8d0fd0d80409c01daff260b249a/image.png)

The **Simple Linear Regression equation** for predicting house prices is:

$$
\hat{Y} = 30 + 0.2X
$$

**Interpretation:**

- **Intercept (30):** A house of size **0 sq ft** (hypothetical) would cost **$30,000.**
- **Slope (0.2):** For every **additional square foot**, the house price increases by **$200.**

## Multiple Linear Regression

Extends simple regression to multiple predictors:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon
$$

### Example: Predicting House Prices with Size & Location

Let’s fit a **Multiple Linear Regression** model with:

- $X_1$ = **House size (sq ft)**
- $X_2$ = **Location Score (1-10)**

```python
# Sample data: House Size (sq ft) & Location Score vs Price ($1000s)
location_score = np.array([7, 8, 6, 9, 5, 7, 6, 8, 7, 9]) # New predictor

# Reshape and stack predictors
X_multi = np.column_stack((house_size, location_score))

# Fit Multiple Linear Regression Model
model_multi = LinearRegression()
model_multi.fit(X_multi, house_price)

# Get regression coefficients
slope_multi = model_multi.coef_ # β1, β2
intercept_multi = model_multi.intercept_ # β0

# Display regression equation
(slope_multi, intercept_multi)
```

```
(array([ 2.00000000e-01, -4.70336726e-16]), np.float64(30.000000000000085))
```

The **Multiple Linear Regression equation** for predicting house prices is: 

$$
\hat{Y} = 30 + 0.2X_1 - 4.7 \times 10^{-16}X_2
$$

**Interpretation:**

- **Size coefficient (0.2):** Each additional square foot increases price by **$200.**
- **Location coefficient** ($\approx 0$): Here, location score has a negligible effect (possibly due to insufficient variation in data).

# Logistic Regression

Used for **binary classification** problems ($Y = 0$ or $Y = 1$). 

## Logistic Function

$$
p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}
$$

### Example: Predicting Customer Churn (Yes/No)

Let’s train a logistic regression model to predict **customer churn** based on **monthly spending.**

```python
from sklearn.linear_model import LogisticRegression

# Sample data: Monthly Spend ($) vs Churn (1 = Yes, 0 = No)
monthly_spend = np.array([20, 25, 30, 35, 40, 50, 60, 70, 80, 90])
churn = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]) # 1 = churn, 0 = no churn

# Reshape for sklearn
X_log = monthly_spend.reshape(-1, 1)
y_log = churn

# Fit Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_log, y_log)

# Get coefficients
slope_log = log_model.coef_[0][0] # β1
intercept_log = log_model.intercept_[0] # β0

# Predict probabilities
x_values = np.linspace(15, 95, 100).reshape(-1, 1)
y_prob = log_model.predict_proba(x_values)[:, 1]

# Plot logistic curve
plt.figure(figsize=(8, 5))
plt.scatter(monthly_spend, churn, color="blue", label="Actual Data")
plt.plot(x_values, y_prob, color="red", linewidth=2, label="Logistic Curve")
plt.xlabel("Monthly Spend ($)")
plt.ylabel("Probability of Churn")
plt.title("Logistic Regression: Monthly Spend vs Churn Probability")
plt.legend()
plt.show()

# Display logistic equation parameters
(slope_log, intercept_log)
```

![image.png](5%20Regression%20and%20Correlation%20Analysis%201b99b8d0fd0d80409c01daff260b249a/image%201.png)

```
(np.float64(-0.7292089775699856), np.float64(27.36246530934355))
```

The **Logistic Regression equation** for predicting customer churn is:

$$
\log\!\biggl(\frac{p}{1 - p}\biggr) = 27.36 - 0.73X
$$

**Interpretation:**

- **Negative slope (-0.73):** As **monthly spend increases**, the **probability of churn decreases.**
- **The curve (shown in red):** Logistic regression models probabilities between **0 and 1**, unlike linear regression.

# Correlation vs. Causation

- **Correlation measures association,** not cause-effect relationships.
- **Causation requires controlled experiments** or **domain knowledge.**

## Example: Correlation Between Ad Spend and Revenue

Let’s compute the **Pearson coefficient.**

```python
# Sample data: Ad Spend ($) vs Revenue ($1000s)
ad_spend = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
revenue = np.array([10, 20, 35, 50, 65, 75, 90, 105, 115, 130])

# Compute Pearson correlation
correlation_coef, _ = stats.pearsonr(ad_spend, revenue)

# Display correlation coefficient
correlation_coef
```

```
np.float64(0.9993139247233791)
```

The **Pearson correlation coefficient** ($r$) between **ad spend and revenue** is **0.999,** indicating a **very strong positive correlation.**

### Key Takeaways

- High correlation ($r \approx 1$) suggests a relationship, but **does not imply causation.**
- **Confounding variables** (e.g., seasonality, competitor actions) might influence both ad spend and revenue.
- To establish causation, you need **controlled experiments** (e.g., A/B testing).

# Summary

| Concept | Explanation |
| --- | --- |
| **Simple Linear Regression** | Models one predictor $X$ and outcome $Y$ with a straight line. |
| **Multiple Linear Regression** | Extends to multiple predictors $X_1,X_2,…$. |
| **Logistic Regression** | Used for **binary classification** (predicts probabilities). |
| **Correlation vs. Causation** | Correlation quantifies association but does not prove cause-effect. |