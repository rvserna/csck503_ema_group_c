"""Random Forest model to predict PM2.5 emissions"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Load the dataset to inspect it
file_path = r"C:\Users\Natalia\OneDrive\Desktop\FGP Machine Learning\final_df.xlsx"
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Check for missing values and column data types
missing_values_summary = data.isnull().sum()
data_types = data.dtypes

# Remove unnecessary columns for the model
columns_to_drop = ['grid_id']
data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

# Impute missing values with mean for numeric columns
numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(data_cleaned[numeric_cols].mean())

# Display basic information about the cleaned dataset
data_cleaned.info()

# Verify the presence of the target variable
assert 'total_pollutant_pm2.5_2019' in data_cleaned.columns, "Target variable not found in the dataset."

# Separate features (X) and target variable (y)
X = data_cleaned.drop(columns=['total_pollutant_pm2.5_2019'])
y = data_cleaned['total_pollutant_pm2.5_2019']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print performance metrics
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

# Get feature importances from the trained model
feature_importances = rf_model.feature_importances_

# Create a DataFrame to sort features by importance
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

# Plot the top 10 important features
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Important Features in Predicting PM2.5 Pollution')
plt.gca().invert_yaxis()
plt.show()
print(importance_df)

# Create a visualization of Actual vs Predicted for PM2.5 emissions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual PM2.5 Values')
plt.ylabel('Predicted PM2.5 Values')
plt.title('Actual vs Predicted PM2.5 Values')
plt.grid(True)
plt.show()