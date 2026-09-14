import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Generate Features
# -----------------------------
def generate_features(n_samples=1000, student_id=12345):

    np.random.seed(student_id % 1000)

    # Feature 1: Normal distribution
    feature_1 = np.random.normal(loc=50, scale=15, size=n_samples)

    # Feature 2: Exponential distribution
    feature_2 = np.random.exponential(scale=2.0, size=n_samples)

    # Add 5 extreme outliers
    feature_2[np.random.choice(n_samples, 5, replace=False)] = \
        np.random.uniform(50, 100, 5)

    return feature_1, feature_2


student_id = 230268

f1, f2 = generate_features(student_id=student_id)

# -----------------------------
# Original Histograms
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(f1, bins=50, color='skyblue')
plt.title("Feature 1")

plt.subplot(1,2,2)
plt.hist(f2, bins=50, color='salmon')
plt.title("Feature 2")

plt.tight_layout()
plt.show()

# -----------------------------
# Standard Scaling
# -----------------------------
scaler = StandardScaler()

f1_scaled = scaler.fit_transform(f1.reshape(-1,1))
f2_scaled = scaler.fit_transform(f2.reshape(-1,1))

# -----------------------------
# Scaled Histograms
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(f1_scaled, bins=50, color='skyblue')
plt.title("Scaled Feature 1")

plt.subplot(1,2,2)
plt.hist(f2_scaled, bins=50, color='salmon')
plt.title("Scaled Feature 2")

plt.tight_layout()
plt.show()

# -----------------------------
# Log Transformation
# -----------------------------
f2_log = np.log1p(f2)

f2_log_scaled = StandardScaler().fit_transform(
    f2_log.reshape(-1,1)
)

plt.figure(figsize=(6,5))
plt.hist(f2_log_scaled, bins=50, color='green')
plt.title("Log + Standard Scaling")
plt.tight_layout()
plt.show()

# -----------------------------
# Load Diabetes Dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

rss = ((y_test - y_pred) ** 2).sum()

n = len(y_test)
p = X_test.shape[1]

rse = (rss / (n - p - 1)) ** 0.5

print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("Residual Standard Error:", rse)

ridge_01 = Ridge(alpha=0.1)
ridge_1 = Ridge(alpha=1.0)
ridge_10 = Ridge(alpha=10.0)

ridge_01.fit(X_train, y_train)
ridge_1.fit(X_train, y_train)
ridge_10.fit(X_train, y_train)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.plot(lr.coef_, marker='o', label='Linear Regression')
plt.plot(ridge_01.coef_, marker='o', label='Ridge α=0.1')
plt.plot(ridge_1.coef_, marker='o', label='Ridge α=1.0')
plt.plot(ridge_10.coef_, marker='o', label='Ridge α=10.0')

plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Comparison of Model Coefficients")
plt.legend()
plt.grid(True)

plt.show()

