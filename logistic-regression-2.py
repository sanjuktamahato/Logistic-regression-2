# Import necessary libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Example dataset (features and target)
X = np.array([[1], [2], [3], [4], [5]])  # Feature values
y = np.array([0, 0, 0, 1, 1])  # Binary target values

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities for the input features
predictions = model.predict_proba(X)[:, 1]  # Get probabilities for class 1

# Print the coefficients (intercept and slope)
print("Intercept (a0):", model.intercept_)
print("Coefficient (a1):", model.coef_)

# Plotting the data points and the decision boundary
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, predictions, color='red', label='Logistic Regression Curve')
plt.xlabel('Feature (x)')
plt.ylabel('Probability of y = 1')
plt.legend()
plt.title('Logistic Regression: Probability vs Feature')
plt.show()

# Example of predicting the probability for a new value of x
x_new = np.array([[6]])
prob_new = model.predict_proba(x_new)[:, 1]
print(f"Predicted probability for x = 6: {prob_new[0]:.4f}")
