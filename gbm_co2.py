"""HistGradientBoostingRegressor model to predict CO2 emissions"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the final dataset
data_path = r"C:\my-project\csck503_ema\csck503_ema_group_c\parsed_data\final_df.pkl"
df = pd.read_pickle(data_path)

# Specify the target variable (CO2 only)
label = "total_pollutant_co2_2019"

# Features (X) will be all columns except the label (CO2)
X = df.drop(columns=[label, 'grid_id'])
y = df[label]

# Split data into 2 sets, training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features and target (standardize to zero mean and unit variance)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 1. Fit and transform the training data
# 2. Transform the test data
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Initialize the model (HistGradientBoostingRegressor)
hgb_regressor = HistGradientBoostingRegressor(random_state=42)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'max_iter': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_leaf': [20, 30]
}

# Optimize hyperparameters using GridSearchCV
grid_search = GridSearchCV(estimator=hgb_regressor, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train_scaled)

# Print the best hyperparameters (identified by GridSearchCV)
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

# Use the best model from GridSearchCV for predictions
best_model = grid_search.best_estimator_

# Fit the model on the training data
best_model.fit(X_train_scaled, y_train_scaled)

# Predict on the scaled test set
y_pred_scaled = best_model.predict(X_test_scaled)

# Inverse transform the predictions and the true values to the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# Calculate performance metrics
mse = mean_squared_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

# Print performance metrics
print(f"Mean Squared Error for CO2: {mse:.2f}")
print(f"R-squared for CO2: {r2:.2f}")

# Create a visualization of Actual vs Predicted for CO2
plt.figure(figsize=(6, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], color='red', lw=2)
plt.xlabel('Actual CO2 Values')
plt.ylabel('Predicted CO2 Values')
plt.title('Actual vs Predicted CO2')
plt.show()