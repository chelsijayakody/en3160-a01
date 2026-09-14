import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --------------------------------------------------
# 1. Generate the dataset
# --------------------------------------------------

indexno = 230268
rng = np.random.default_rng(indexno)

X = rng.uniform(-3, 3, size=(160, 1))

y = 2.5 + 1.8 * X[:, 0] + rng.normal(0, 0.8, size=160)

# Add large errors to 12 observations
outlier_idx = rng.choice(len(y), size=12, replace=False)
y[outlier_idx] += rng.uniform(10, 15, size=12)

# --------------------------------------------------
# 2. 80:20 train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 3. Fit Ordinary Least Squares model
# --------------------------------------------------

ols = LinearRegression()
ols.fit(X_train, y_train)

# Predictions
y_train_pred = ols.predict(X_train)
y_test_pred = ols.predict(X_test)

# --------------------------------------------------
# 4. Calculate results
# --------------------------------------------------

w0 = ols.intercept_
w1 = ols.coef_[0]

MSE_train = mean_squared_error(y_train, y_train_pred)
MSE_test = mean_squared_error(y_test, y_test_pred)

print("Intercept (w0):", w0)
print("Slope (w1):", w1)
print("Training MSE:", MSE_train)
print("Test MSE:", MSE_test)

# --------------------------------------------------
# 5. Plot data and fitted regression line
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    X_train,
    y_train,
    label="Training data"
)

plt.scatter(
    X_test,
    y_test,
    label="Test data"
)

# Generate points for regression line
x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_line = ols.predict(x_line)

plt.plot(
    x_line,
    y_line,
    label="OLS regression line"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("OLS Regression with Vertical Outliers")
plt.legend()
plt.grid(True)

plt.show()

# --------------------------------------------------
# Weighted Least Squares
# --------------------------------------------------

# Create the design matrix
# First column = 1 for the intercept
X_design = np.column_stack((np.ones(len(X_train)), X_train[:, 0]))

# Create weights
weights = np.ones(len(y_train))

# Identify which training observations are outliers
# outlier_idx contains indices relative to the original dataset
train_indices = np.arange(len(y))[X_train.shape[0]*0:]  # not used directly

# We need to identify outliers belonging to the training set.
# Using the actual train/test indices is safer:
X_train2, X_test2, y_train2, y_test2, idx_train, idx_test = train_test_split(
    X,
    y,
    np.arange(len(y)),
    test_size=0.20,
    random_state=42
)

# Rebuild design matrix
X_design = np.column_stack(
    (np.ones(len(X_train2)), X_train2[:, 0])
)

# Assign weights
weights = np.ones(len(y_train2))

# Outliers in the training set receive weight 0.05
for i in range(len(y_train2)):
    if idx_train[i] in outlier_idx:
        weights[i] = 0.05

# Construct diagonal weighting matrix
A = np.diag(weights)

# Calculate WLS parameters
w_wls = np.linalg.solve(
    X_design.T @ A @ X_design,
    X_design.T @ A @ y_train2
)

w0_wls = w_wls[0]
w1_wls = w_wls[1]

print("WLS intercept:", w0_wls)
print("WLS slope:", w1_wls)


# --------------------------------------------------
# Compare OLS and WLS
# --------------------------------------------------

plt.figure(figsize=(8, 5))

# Plot all observations
plt.scatter(
    X,
    y,
    label="Data"
)

# Generate points for regression lines
x_line = np.linspace(X.min(), X.max(), 200)

# OLS prediction
y_ols_line = ols.intercept_ + ols.coef_[0] * x_line

# WLS prediction
y_wls_line = w0_wls + w1_wls * x_line

# Plot OLS
plt.plot(
    x_line,
    y_ols_line,
    label="OLS"
)

# Plot WLS
plt.plot(
    x_line,
    y_wls_line,
    label="WLS"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Comparison of OLS and Weighted Least Squares")
plt.legend()
plt.grid(True)

plt.show()