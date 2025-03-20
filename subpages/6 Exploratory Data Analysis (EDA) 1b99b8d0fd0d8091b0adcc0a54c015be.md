# 6. Exploratory Data Analysis (EDA)

# Identifying Missing Values and Outliers

## Checking for Missing Values

In Python, missing values are usually represented as `NaN` (Not a Number). You can check for them using:

```python
df.isnull().sum()
```

## Example: Checking Missing Values in a Dataset

Let’s create a sample dataset with some missing values and analyze them.

`data`

| `Customer_Id` | `Age` | `Salary` | `City` |
| --- | --- | --- | --- |
| 101 | 25 | 30000 | NY |
| 102 | 30 | 35000 | LA |
| 103 | 35 | 40000 | SF |
| 104 | np.nan | 45000 | NY |
| 105 | 40 | np.nan | LA |
| 106 | 45 | 5000 | np.nan |
| 107 | np.nan | 250000 | SF |

```python
# Sample dataset with missing values
import pandas as pd
import numpy as np

data = {
    "Customer_ID": [101, 102, 103, 104, 105, 106, 107],
    "Age": [25, 30, 35, np.nan, 40, 45, np.nan],
    "Salary": [30000, 35000, 40000, 45000, np.nan, 50000, 250000],
    "City": ["NY", "LA", "SF", "NY", "LA", np.nan, "SF"]
}

df = pd.DataFrame(data)

# Check for missing values
missing_values = df.isnull().sum()

# Display dataset and missing values
missing_values
```

```
Customer_ID    0
Age            2
Salary         1
City           1
dtype: int64
```

The dataset contains missing values in the **Age, Salary,** and **City** columns. You can handle these missing values using different strategies:

## Handling Missing Values

### Drop Missing Values

If few and non-critical:

```python
df.dropna(inplace=True)
```

### Fill with mean/median

For numerical data:

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
```

### Fill with mode

For categorical data:

```python
df['City'].fillna(df['City'].mode()[0], inplace=True)
```

## Detecting Outliers

Outliers are extreme values that deviate significantly from the rest of the data. They can be detected using:

- **Boxplots** (Graphical)
- **Z-score method** ($|Z| > 3$ suggests an outlier)
- **IQR method** ($Outlier = Q_1 - 1.5 \times IQR$ or Q_3 + 1.5 \times IQR)

Let’s find **outliers** in **Salary** using a boxplot.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot for detecting outliers in Salary
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["Salary"])
plt.title("Boxplot of Salary (Detecting Outliers)")
plt.ylabel("Salary")
plt.show()
```

![image.png](6%20Exploratory%20Data%20Analysis%20(EDA)%201b99b8d0fd0d8091b0adcc0a54c015be/image.png)

The **boxplot** helps visualize outliers - any point **outside the whiskers** are potential outliers.

To **numerically detect outliers**, we use the **IQR method**:

$$
\text{Outlier} = Q1 - 1.5 \times \mathrm{IQR} \quad \text{or} \quad Q3 + 1.5 \times \mathrm{IQR}
$$

Let’s compute this for **Salary.**

```python
# Compute Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)]

# Display outliers
outliers
```

|  | Customer_ID | Age | Salary | City |
| --- | --- | --- | --- | --- |
| 6 | 107 | NaN | 250000.0 | SF |

It’s always a good practice to verify with domain knowledge - extreme values might still be valid data points rather than errors.

# Feature Engineering and Transformation

Feature engineering involves **creating new variables** to improve models.

## Examples of Feature Engineering

### Extracting “Year” from a Date Column

```python
df["Year"] = df["Date"].dt.year
```

### Creating Interaction Features (Multiplying Variables)

```python
df["Income_Per_Age"] = df["Salary"] / df["Age"]
```

### Log Transformation to Normalize Skewed Data

```python
df["Log_Salary"] = np.log(df["Salary"])
```

### Example: Log Transformation

Let’s apply **log transformation** to the **Salary** column and visualize the effect.

```python
# Apply log transformation to Salary
df["Log_Salary"] = np.log(df["Salary"])
# Plot original vs log-transformed Salary distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df["Salary"].dropna(), bins=10, kde=True)
plt.title("Original Salary Distribution")
plt.subplot(1, 2, 2)
sns.histplot(df["Log_Salary"].dropna(), bins=10, kde=True, color="green")
plt.title("Log-Transformed Salary Distribution")

```

![image.png](6%20Exploratory%20Data%20Analysis%20(EDA)%201b99b8d0fd0d8091b0adcc0a54c015be/image%201.png)

The **log transformation** makes the salary distribution more **normally distributed,** which is useful for **reducing skewness** and improving the performance of statistical models.

# Data Cleaning and Preprocessing

Data cleaning ensures that raw data is **structured, consistent,** and **ready for analysis.**

## Common Data Cleaning Tasks

### Removing Duplicate Records

```python
df.drop_duplicates(inplace=True)
```

### Standardizing Categorical Variables

```python
df["City"] = df["City"].str.upper()
```

### Handling Missing Values (as discussed earlier)

### Scaling Numerical Features (e.g., for ML models)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df["Salary_Scaled"] = scaler.fit_transform(df[["Salary"]])
```

# Summary

| Task | Description |
| --- | --- |
| **Missing Values** | Identified using `.isnull().sum()` and handled via deletion or imputation. |
| **Outliers** | Detected using **boxplots, IQR,** or **Z-score methods.** |
| **Feature Engineering** | Creating new variables (e.g., log transformations, date extractions). |
| **Data Cleaning** | Removing duplicates, standardizing categories, fixing inconsistencies. |
| **Scaling Features** | Normalizing numerical variables for ML models. |