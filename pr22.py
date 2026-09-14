# ============================================================
# EN3150 Assignment - Logistic Regression
# Questions 2 and 3
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# QUESTION 2 - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 2 - LOGISTIC REGRESSION")
print("=" * 60)


# ------------------------------------------------------------
# 2.1 DATA GENERATION
# ------------------------------------------------------------

X_num, y = make_classification(
    n_samples=600,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    class_sep=0.9,
    weights=[0.68, 0.32],
    flip_y=0.03,
    random_state=21
)

df = pd.DataFrame(
    X_num,
    columns=["x1", "x2", "x3", "x4"]
)

rng = np.random.default_rng(21)

df["region"] = rng.choice(
    ["north", "south", "east"],
    size=len(df)
)

df["device"] = rng.choice(
    ["mobile", "desktop"],
    size=len(df)
)

# Add 25 missing values to x3
df.loc[
    rng.choice(len(df), 25, replace=False),
    "x3"
] = np.nan

print("\nDataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(pd.Series(y).value_counts())


# ------------------------------------------------------------
# 2.2 TRAIN-TEST SPLIT
# ------------------------------------------------------------

X = df

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 2.3 PREPROCESSING PIPELINE
# ------------------------------------------------------------

num_cols = [
    "x1",
    "x2",
    "x3",
    "x4"
]

cat_cols = [
    "region",
    "device"
]

# Numerical preprocessing:
# 1. Median imputation
# 2. Standardization

num_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

# Categorical preprocessing:
# 1. Most frequent imputation
# 2. One-hot encoding

cat_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# Combine numerical and categorical preprocessing

preprocessor = ColumnTransformer([
    (
        "num",
        num_pipe,
        num_cols
    ),
    (
        "cat",
        cat_pipe,
        cat_cols
    )
])


# Logistic Regression model

pipe = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            solver="liblinear",
            max_iter=1000
        )
    )
])


# ------------------------------------------------------------
# 2.4 GRID SEARCH
# ------------------------------------------------------------

param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "classifier__penalty": [
        "l1",
        "l2"
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    pipe,
    param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

print("\nRunning Grid Search...")

grid_search.fit(
    X_train,
    y_train
)


print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1:")
print(
    f"{grid_search.best_score_:.6f}"
)


# ------------------------------------------------------------
# 2.5 TEST SET EVALUATION
# ------------------------------------------------------------

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "-" * 40)
print("TEST SET PERFORMANCE")
print("-" * 40)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print("\nConfusion Matrix:")
print(cm)


# Plot confusion matrix

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.show()


# ------------------------------------------------------------
# 2.6 DECISION THRESHOLD = 0.35
# ------------------------------------------------------------

y_prob = best_model.predict_proba(
    X_test
)[:, 1]

threshold_035 = 0.35

y_pred_035 = (
    y_prob >= threshold_035
).astype(int)

precision_035 = precision_score(
    y_test,
    y_pred_035
)

recall_035 = recall_score(
    y_test,
    y_pred_035
)

print("\n" + "-" * 40)
print("THRESHOLD = 0.35")
print("-" * 40)

print(
    f"Precision : {precision_035:.4f}"
)

print(
    f"Recall    : {recall_035:.4f}"
)


print("\nPrecision-Recall comparison:")

print(
    f"Threshold 0.50 -> "
    f"Precision = {precision:.4f}, "
    f"Recall = {recall:.4f}"
)

print(
    f"Threshold 0.35 -> "
    f"Precision = {precision_035:.4f}, "
    f"Recall = {recall_035:.4f}"
)


# ============================================================
# 2.7 DIRECT LOGISTIC REGRESSION PROBABILITY
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION PROBABILITY CALCULATION")
print("=" * 60)

x1 = 45
x2 = 3.4

z = (
    -7.2
    + 0.08 * x1
    + 1.4 * x2
)

p = 1 / (
    1 + np.exp(-z)
)

print("\nFor x1 = 45 and x2 = 3.4:")

print(
    f"z = {z:.4f}"
)

print(
    f"Probability = {p:.4f}"
)

print(
    f"Probability (%) = {p * 100:.2f}%"
)


# ------------------------------------------------------------
# Required study hours for p = 0.70
# ------------------------------------------------------------

target_p = 0.70
x2 = 3.4

target_z = np.log(
    target_p / (1 - target_p)
)

x1_required = (
    target_z
    + 7.2
    - 1.4 * x2
) / 0.08

print(
    "\nRequired study hours for p = 0.70:"
)

print(
    f"x1 = {x1_required:.4f} hours"
)


# ============================================================
# QUESTION 3 - OPTIMIZING LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 3 - OPTIMIZING LOGISTIC REGRESSION")
print("=" * 60)


# ------------------------------------------------------------
# 3.1 NUMERICAL DATA FOR MANUAL OPTIMIZATION
# ------------------------------------------------------------

# Use only numerical features for the manual
# optimization algorithms.

X_num_df = df[
    ["x1", "x2", "x3", "x4"]
].copy()

# Split using the same indices as the original
# train-test split

X_train_num = X_num_df.loc[
    X_train.index
].copy()

X_test_num = X_num_df.loc[
    X_test.index
].copy()


# ------------------------------------------------------------
# Handle missing values
# ------------------------------------------------------------

median_values = X_train_num.median()

X_train_num = X_train_num.fillna(
    median_values
)

X_test_num = X_test_num.fillna(
    median_values
)


# ------------------------------------------------------------
# Standardize using TRAINING DATA ONLY
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_num
)

X_test_scaled = scaler.transform(
    X_test_num
)


# ------------------------------------------------------------
# Add bias/intercept column
# ------------------------------------------------------------

X_train_aug = np.column_stack([
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
])

X_test_aug = np.column_stack([
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
])

print("\nTraining augmented shape:")
print(X_train_aug.shape)

print("\nTesting augmented shape:")
print(X_test_aug.shape)


# ============================================================
# 3.2 NUMERICALLY STABLE SIGMOID
# ============================================================

def sigmoid(z):

    result = np.empty_like(
        z,
        dtype=float
    )

    positive = z >= 0
    negative = ~positive

    # Positive values
    result[positive] = 1 / (
        1 + np.exp(
            -z[positive]
        )
    )

    # Negative values
    exp_z = np.exp(
        z[negative]
    )

    result[negative] = (
        exp_z /
        (1 + exp_z)
    )

    return result


# ============================================================
# 3.3 NUMERICALLY STABLE CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(
    w,
    X,
    y
):

    z = X @ w

    loss = (
        np.logaddexp(0, z)
        - y * z
    )

    return np.mean(loss)


# ============================================================
# 3.4 GRADIENT
# ============================================================

def compute_gradient(
    w,
    X,
    y
):

    N = len(y)

    p = sigmoid(
        X @ w
    )

    gradient = (
        X.T @ (p - y)
    ) / N

    return gradient


# ============================================================
# 3.5 BATCH GRADIENT DESCENT
# ============================================================

def batch_gradient_descent(
    X,
    y,
    learning_rate=0.1,
    max_iterations=300,
    tolerance=1e-8
):

    # Initialize weights
    w = np.zeros(
        X.shape[1]
    )

    losses = []

    # Initial loss
    current_loss = cross_entropy_loss(
        w,
        X,
        y
    )

    losses.append(
        current_loss
    )

    for iteration in range(
        1,
        max_iterations + 1
    ):

        # Compute gradient
        gradient = compute_gradient(
            w,
            X,
            y
        )

        # Update weights
        w = (
            w
            - learning_rate * gradient
        )

        # Compute new loss
        new_loss = cross_entropy_loss(
            w,
            X,
            y
        )

        losses.append(
            new_loss
        )

        # Early stopping
        if abs(
            new_loss - current_loss
        ) < tolerance:

            break

        current_loss = new_loss

    return (
        w,
        losses,
        iteration
    )


# ------------------------------------------------------------
# Train BGD
# ------------------------------------------------------------

w_bgd, bgd_losses, bgd_iterations = (
    batch_gradient_descent(
        X_train_aug,
        y_train,
        learning_rate=0.1,
        max_iterations=300,
        tolerance=1e-8
    )
)


# ------------------------------------------------------------
# Evaluate BGD
# ------------------------------------------------------------

test_prob_bgd = sigmoid(
    X_test_aug @ w_bgd
)

y_pred_bgd = (
    test_prob_bgd >= 0.5
).astype(int)

test_accuracy_bgd = np.mean(
    y_pred_bgd == y_test
)


print("\n" + "-" * 40)
print("BATCH GRADIENT DESCENT")
print("-" * 40)

print("\nFinal weights:")

print(w_bgd)

print(
    "\nIterations:",
    bgd_iterations
)

print(
    "Test Accuracy:",
    f"{test_accuracy_bgd:.4f}"
)


# ============================================================
# 3.6 MINI-BATCH GRADIENT DESCENT
# ============================================================

def mini_batch_gradient_descent(
    X,
    y,
    batch_size=32,
    learning_rate=0.05,
    max_epochs=100
):

    w = np.zeros(
        X.shape[1]
    )

    losses = []

    N = len(y)

    for epoch in range(
        max_epochs
    ):

        # Shuffle every epoch
        indices = np.random.permutation(
            N
        )

        X_shuffled = X[
            indices
        ]

        y_shuffled = y[
            indices
        ]

        # Mini-batches
        for start in range(
            0,
            N,
            batch_size
        ):

            end = start + batch_size

            X_batch = X_shuffled[
                start:end
            ]

            y_batch = y_shuffled[
                start:end
            ]

            # Gradient
            gradient = compute_gradient(
                w,
                X_batch,
                y_batch
            )

            # Update
            w = (
                w
                - learning_rate * gradient
            )

        # Full training loss
        epoch_loss = cross_entropy_loss(
            w,
            X,
            y
        )

        losses.append(
            epoch_loss
        )

    return (
        w,
        losses
    )


# ------------------------------------------------------------
# Train Mini-Batch GD
# ------------------------------------------------------------

np.random.seed(42)

w_mbgd, mbgd_losses = (
    mini_batch_gradient_descent(
        X_train_aug,
        y_train,
        batch_size=32,
        learning_rate=0.05,
        max_epochs=100
    )
)


# ------------------------------------------------------------
# Evaluate Mini-Batch GD
# ------------------------------------------------------------

test_prob_mbgd = sigmoid(
    X_test_aug @ w_mbgd
)

y_pred_mbgd = (
    test_prob_mbgd >= 0.5
).astype(int)

test_accuracy_mbgd = np.mean(
    y_pred_mbgd == y_test
)


print("\n" + "-" * 40)
print("MINI-BATCH GRADIENT DESCENT")
print("-" * 40)

print("\nFinal weights:")

print(w_mbgd)

print(
    "\nTest Accuracy:",
    f"{test_accuracy_mbgd:.4f}"
)


# ============================================================
# 3.7 HESSIAN
# ============================================================

def compute_hessian(
    w,
    X,
    y
):

    N = len(y)

    p = sigmoid(
        X @ w
    )

    S = p * (
        1 - p
    )

    H = (
        X.T
        @ (
            X * S[:, None]
        )
    ) / N

    return H


# ============================================================
# 3.8 DAMPED NEWTON'S METHOD
# ============================================================

def newton_method(
    X,
    y,
    max_iterations=20,
    tolerance=1e-8
):

    # Initialize weights
    w = np.zeros(
        X.shape[1]
    )

    losses = []

    # Initial loss
    current_loss = cross_entropy_loss(
        w,
        X,
        y
    )

    losses.append(
        current_loss
    )

    for iteration in range(
        1,
        max_iterations + 1
    ):

        # Gradient
        gradient = compute_gradient(
            w,
            X,
            y
        )

        # Hessian
        H = compute_hessian(
            w,
            X,
            y
        )

        # Damping
        damping = 1e-6

        H_damped = (
            H
            + damping
            * np.eye(
                X.shape[1]
            )
        )

        # Solve H*d = gradient
        direction = np.linalg.solve(
            H_damped,
            gradient
        )

        # Newton update
        w = w - direction

        # New loss
        new_loss = cross_entropy_loss(
            w,
            X,
            y
        )

        losses.append(
            new_loss
        )

        # Early stopping
        if abs(
            new_loss - current_loss
        ) < tolerance:

            break

        current_loss = new_loss

    return (
        w,
        losses,
        iteration
    )


# ------------------------------------------------------------
# Train Newton's Method
# ------------------------------------------------------------

w_newton, newton_losses, newton_iterations = (
    newton_method(
        X_train_aug,
        y_train,
        max_iterations=20,
        tolerance=1e-8
    )
)


# ------------------------------------------------------------
# Evaluate Newton
# ------------------------------------------------------------

test_prob_newton = sigmoid(
    X_test_aug @ w_newton
)

y_pred_newton = (
    test_prob_newton >= 0.5
).astype(int)

test_accuracy_newton = np.mean(
    y_pred_newton == y_test
)


print("\n" + "-" * 40)
print("DAMPED NEWTON'S METHOD")
print("-" * 40)

print("\nFinal weights:")

print(w_newton)

print(
    "\nIterations:",
    newton_iterations
)

print(
    "Test Accuracy:",
    f"{test_accuracy_newton:.4f}"
)


# ============================================================
# 3.9 COMPARE BGD AND NEWTON'S METHOD
# ============================================================

print("\n" + "=" * 60)
print("BGD vs NEWTON COMPARISON")
print("=" * 60)

print(
    "\nBGD iterations:",
    bgd_iterations
)

print(
    "Newton iterations:",
    newton_iterations
)

print(
    "\nBGD test accuracy:",
    f"{test_accuracy_bgd:.4f}"
)

print(
    "Newton test accuracy:",
    f"{test_accuracy_newton:.4f}"
)


# ------------------------------------------------------------
# Plot training loss
# ------------------------------------------------------------

plt.figure()

plt.plot(
    bgd_losses,
    label="Batch Gradient Descent"
)

plt.plot(
    newton_losses,
    label="Newton's Method"
)

plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(
    "Batch Gradient Descent vs Newton's Method"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 3.10 MINI-BATCH GD VS BATCH GD
# ============================================================

plt.figure()

plt.plot(
    bgd_losses,
    label="Batch Gradient Descent"
)

plt.plot(
    mbgd_losses,
    label="Mini-Batch Gradient Descent"
)

plt.xlabel("Iteration / Epoch")
plt.ylabel("Training Loss")

plt.title(
    "Mini-Batch GD vs Batch GD"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 3.11 EFFECT OF FEATURE SCALING
# ============================================================

print("\n" + "=" * 60)
print("EFFECT OF FEATURE SCALING")
print("=" * 60)


# Make a copy of standardized training data
X_bad = X_train_scaled.copy()

# Multiply second feature by 100
X_bad[:, 1] = (
    X_bad[:, 1] * 100
)

# Add bias
X_bad_aug = np.column_stack([
    np.ones(X_bad.shape[0]),
    X_bad
])


# ------------------------------------------------------------
# Calculate Hessian at initial weights
# ------------------------------------------------------------

w_initial = np.zeros(
    X_bad_aug.shape[1]
)

H_bad = compute_hessian(
    w_initial,
    X_bad_aug,
    y_train
)


# Eigenvalues
eigenvalues = np.linalg.eigvalsh(
    H_bad
)

# Remove extremely small numerical values
positive_eigenvalues = (
    eigenvalues[
        eigenvalues > 1e-12
    ]
)

if len(positive_eigenvalues) > 0:

    condition_number = (
        np.max(
            positive_eigenvalues
        )
        /
        np.min(
            positive_eigenvalues
        )
    )

else:

    condition_number = np.inf


print(
    "\nCondition number of Hessian:"
)

print(
    condition_number
)


# ------------------------------------------------------------
# Run BGD with badly scaled feature
# ------------------------------------------------------------

w_bad, bad_losses, bad_iterations = (
    batch_gradient_descent(
        X_bad_aug,
        y_train,
        learning_rate=0.1,
        max_iterations=300,
        tolerance=1e-8
    )
)


print(
    "\nIterations with badly scaled feature:",
    bad_iterations
)


# ------------------------------------------------------------
# Plot effect of poor scaling
# ------------------------------------------------------------

plt.figure()

plt.plot(
    bad_losses,
    label="Unstandardized Feature"
)

plt.xlabel("Iteration")
plt.ylabel("Training Loss")

plt.title(
    "Effect of Feature Scaling on Gradient Descent"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    "\n--- Question 2 ---"
)

print(
    "Best parameters:",
    grid_search.best_params_
)

print(
    "Best CV F1:",
    f"{grid_search.best_score_:.6f}"
)

print(
    "Test Accuracy:",
    f"{accuracy:.4f}"
)

print(
    "Test Precision:",
    f"{precision:.4f}"
)

print(
    "Test Recall:",
    f"{recall:.4f}"
)

print(
    "Test F1:",
    f"{f1:.4f}"
)

print(
    "\nThreshold 0.35:"
)

print(
    "Precision:",
    f"{precision_035:.4f}"
)

print(
    "Recall:",
    f"{recall_035:.4f}"
)


print(
    "\n--- Question 3 ---"
)

print(
    "BGD Accuracy:",
    f"{test_accuracy_bgd:.4f}"
)

print(
    "Mini-Batch GD Accuracy:",
    f"{test_accuracy_mbgd:.4f}"
)

print(
    "Newton Accuracy:",
    f"{test_accuracy_newton:.4f}"
)

print(
    "\nBGD iterations:",
    bgd_iterations
)

print(
    "Newton iterations:",
    newton_iterations
)

print("\nDone!")

