# This is a Prototype Python script designed to model and visualize the trend in bag production over time using polynomial regression.
# It utilizes the NumPy library for numerical computations and Matplotlib for plotting the data and the regression model.

import matplotlib.pyplot as plt
import numpy as np  

# Defining the columns and rows data. Columns represent different metrics or features, and Rows represent production data over time.
Columns = ['Data1', 'Data2', 'Data3', 'Data144']  # Placeholder for actual column names
Rows = [1863, 1614, 2570, 17689]  # Production data over time

# Converting the time series data into NumPy arrays for processing
x = np.array(range(1, len(Rows) + 1), dtype=float)  # Time steps as x-axis (1 to number of data points)
y = np.array(Rows, dtype=float)  # Production data as y-axis

# Function to generate polynomial features for the model. It transforms the input x into its polynomial terms up to the specified degree.
def generate_normal_polynomial_features(x, degree):
    X_poly_normal = np.ones((len(x), degree + 1), dtype=float)  # Initializing the feature matrix with ones for the bias term
    for i in range(1, degree + 1):
        X_poly_normal[:, i] = x ** i  # Generating polynomial features
    return X_poly_normal

# Degree Testing
degree = 1  # It seems this is a lienar model
degree = 3  # Degree of the polynomial model

# Generating polynomial features based on the specified degree
X_poly_normal = generate_normal_polynomial_features(x, degree)

# Calculating the model parameters (theta) using the Normal Equation method
theta_normal = np.linalg.inv(X_poly_normal.T.dot(X_poly_normal)).dot(X_poly_normal.T).dot(y)

# Predicting the production using the polynomial model
y_pred_normal = X_poly_normal.dot(theta_normal)

# Plotting the actual production data and the polynomial regression model to visualize the trend
plt.scatter(x, y, color='blue', label='Actual Production')  # Plotting the actual data points
plt.plot(x, y_pred_normal, color='green', label='Polynomial Regression Model')  # Plotting the regression model
plt.title('Trend in Bag\'s Production with Logarithmic Polynomial Regression')  # Setting the title of the plot
plt.xlabel('Time (Month)')  # Label for the x-axis
plt.ylabel('Production')  # Label for the y-axis
plt.legend()  # Displaying the legend to identify the data and the model

plt.show()  # Displaying the plot