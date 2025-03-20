# 2. Descriptive Statistics

# Measures of Central Tendency

These measures summarize the center of a dataset:

## **Mean**

($\mu$ **or** $\bar{x}$) - The arithmetic average

$$
\bar{x} = \frac{\sum x_i}{n}
$$

## **Median**

The middle value when sorted.

## **Mode**

The most frequent occurring value.

## Example: Central Tendency Calculation

**Dataset:** $[2, 5, 5, 6, 8]$

### **Mean Calculation**

$$
\bar{x} = \frac{2 + 5 + 5 + 6 + 8}{5} = \frac{26}{5} = 5.2
$$

### **Median Calculation**

- Sorted data: $[2, 5, 5, 6, 8]$
- Middle value: **5** (since it’s the third value in a 5-number list)

### **Mode**

The most frequent value is **5** (appears twice).

# Measures of Dispersion

These describe how spread out the data is. 

## Range

$$
\text{Range} = \max(x) - \min(x)
$$

## Variance

$$
\sigma^2 = \frac{\sum (x_i - \bar{x})^2}{n}
$$

## Standard Deviation

$$
\sigma = \sqrt{\sigma^2}
$$

## Interquartile Range (IQR)

$$
\text{IQR} = Q_3 - Q_1
$$

## Example: Dispersion Calculation

**Dataset:** [2, 5, 5, 6, 8]

- **Range:** 8 - 2 = 6
- **Variance & Standard Deviation Calculation (Using Python for output):**

Computing in Python:

```python
import numpy as np
import pandas as pd

# Given dataset
data = np.array([2, 5, 5, 6, 8])

# Measures of Dispersion
range_value = np.max(data) - np.min(data)
variance = np.var(data, ddof=0)  # Population variance
std_dev = np.sqrt(variance)  # Standard deviation
iqr = np.percentile(data, 75) - np.percentile(data, 25)  # IQR

# Create DataFrame to display results
dispersion_metrics = pd.DataFrame({"Measure": ["Range", "Variance", "Standard Deviation", "Interquartile Range (IQR)"],
																	 "Value": [range_value, variance, std_dev, iqr]})
print(dispersion_metrics)																	
```

```
                     Measure     Value
0                      Range  6.000000
1                   Variance  3.760000
2         Standard Deviation  1.939072
3  Interquartile Range (IQR)  1.000000
```