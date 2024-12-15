import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_utils import remove_outliers_by_iqr

def rf_pollution(pollutant, include_other_pollutants=False, remove_outliers=False):
    data = pd.read_pickle("./parsed_data/final_df.pkl")

    # Drop any irrelevant features
    data.drop(columns="grid_id", inplace=True)

    # Specify the target variable
    label = f"total_pollutant_{pollutant}_2019"

    # Remove outliers if appropriate
    if remove_outliers:
        data = remove_outliers_by_iqr(data, [label])

    # Impute missing values with mean for numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Split features into train and label based on pollutant and include_other_pollutants
    if include_other_pollutants:
        X = data[[col for col in data.columns if col != label]]
    else:
        X = data[[col for col in data.columns if not col.startswith("total_pollutant")]]
    y = data[label]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print performance metrics
    print("Mean Absolute Error:", mae)
    print("Mean Square Error: ", mse)
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
    plt.title(f'Top 10 Important Features in Predicting {pollutant} Pollution')
    plt.gca().invert_yaxis()
    plt.show()
    print(importance_df)

    # Create a visualization of Actual vs Predicted for CO2 emissions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    plt.xlabel(f'Actual {pollutant} Values')
    plt.ylabel(f'Predicted {pollutant} Values')
    plt.title(f'Actual vs Predicted {pollutant} Values')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    for pollutant in ("pm10", "pm2.5", "co2", "nox"):
        rf_pollution(pollutant, remove_outliers=True)