"""Random Forest model to predict PM10 emissions"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Load the dataset to inspect it
file_path = 'C:/Users/Natalia/OneDrive/Desktop/FGP Machine Learning/final_df.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Check for missing values and column data types
missing_values_summary = data.isnull().sum()
data_types = data.dtypes

# Remove unnecessary columns for the model
columns_to_drop = ['geometry','Borough', 'Zone']
data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

# Impute missing values with mean for numeric columns
numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(data_cleaned[numeric_cols].mean())

# Display basic information about the cleaned dataset
data_cleaned.info()

# Verify the presence of the target variable
assert 'total_pollutant_pm10_2019' in data_cleaned.columns, "Target variable not found in the dataset."

# Separate features (X) and target variable (y)
X = data_cleaned.drop(columns=['total_pollutant_pm10_2019'])
y = data_cleaned['total_pollutant_pm10_2019']

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

# Create a visualization of Actual vs Predicted for NOX emissions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual PM10 Values')
plt.ylabel('Predicted PM10 Values')
plt.title('Actual vs Predicted PM10 Values')
plt.grid(True)
plt.show()