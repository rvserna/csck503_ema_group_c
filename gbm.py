import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import GridEmissions

# create instance of GridEmissions, extract features and labels
grid = GridEmissions()
X, y = grid.get_features_and_labels()

# split data into 2 sets, training and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize and train the Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

gbm_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_regressor.fit(X_train, y_train)

# predict on the test set
y_pred = gbm_regressor.predict(X_test)

# evaluate the model using Mean Squared Error and R-squared
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# visualizing the model's performance

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs Predicted values')
plt.show()

