import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = {
    'Sqft': [1500, 2000, 1200, 2500, 1800],
    'Beds': [3, 4, 2, 4, 3],
    'Price': [300000, 450000, 250000, 550000, 380000]
}

df = pd.DataFrame(data)
df.to_csv("house_prices.csv", index=False)
X = df[['Sqft', 'Beds']].values
y = df[['Price']].values
x_max = X.max(axis=0)
y_max = y.max()
X_scaled = X / x_max
y_scaled = y / y_max

# Initialize weights
np.random.seed(42)
w_h = np.random.uniform(size=(2, 4))
w_o = np.random.uniform(size=(4, 1))

lr = 0.1
loss_history = []

print("Training the model... Please wait.")

# Training loop
for epoch in range(5000):
    hidden_output = 1 / (1 + np.exp(-np.dot(X_scaled, w_h)))
    prediction_scaled = np.dot(hidden_output, w_o)

    error = y_scaled - prediction_scaled
    avg_loss = np.mean(np.abs(error))
    loss_history.append(avg_loss)

    d_output = error
    error_hidden = d_output.dot(w_o.T)
    d_hidden = error_hidden * (hidden_output * (1 - hidden_output))

    w_o += hidden_output.T.dot(d_output) * lr
    w_h += X_scaled.T.dot(d_hidden) * lr

# Final predictions
final_prices = prediction_scaled * y_max
print("\n--- Final House Price Predictions ---")
for i in range(len(X)):
    print(f"Actual: {int(y[i][0])} | Predicted: {int(final_prices[i][0])}")
# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("average Error")
plt.title("House Price Model")
plt.grid(True)
plt.show()
