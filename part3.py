import numpy as np

# Dataset
x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([2.5, 4.1, 6.2, 8.0, 35.0, 12.1], dtype=float)

# Predictions
y_pred_A = 3.8 * x - 2
y_pred_B = 2.0 * x + 0.2

print("Model A predictions:")
print(y_pred_A)

print("\nModel B predictions:")
print(y_pred_B)

mse_A = np.mean((y - y_pred_A) ** 2)
mse_B = np.mean((y - y_pred_B) ** 2)

print("\nMSE Model A =", mse_A)
print("MSE Model B =", mse_B)
def huber_loss(y_true, y_pred, delta=3):
    error = y_true - y_pred
    abs_error = np.abs(error)

    loss = np.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * abs_error - 0.5 * delta**2
    )

    return np.mean(loss)

huber_A = huber_loss(y, y_pred_A, delta=3)
huber_B = huber_loss(y, y_pred_B, delta=3)

print("\nHuber Loss Model A =", huber_A)
print("Huber Loss Model B =", huber_B)