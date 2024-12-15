import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ml_utils import remove_outliers_by_iqr

def gbm_pollution(pollutant, include_other_pollutants=False, remove_outliers=False):
    df = pd.read_pickle("./parsed_data/final_df.pkl")

    # Drop any irrelevant features
    df.drop(columns="grid_id", inplace=True)

    # Specify the target variable
    label = f"total_pollutant_{pollutant}_2019"

    # Remove outliers if appropriate
    if remove_outliers:
        df = remove_outliers_by_iqr(df, [label])

    # Split features into test and train based on label and include_other_pollutants
    if include_other_pollutants:
        X = df[[col for col in df.columns if col != label]]
    else:
        X = df[[col for col in df.columns if not col.startswith("total_pollutant")]]
    
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
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print performance metrics
    print(f"Mean Absolute Error for {pollutant}: {mae:.2f}")
    print(f"Mean Squared Error for {pollutant}: {mse:.2f}")
    print(f"R-squared for {pollutant}: {r2:.2f}")

    # Create a visualization of Actual vs Predicted for CO2
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_actual, y_pred, alpha=0.5)
    plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], color='red', lw=2)
    plt.xlabel(f'Actual {pollutant} Values')
    plt.ylabel(f'Predicted {pollutant} Values')
    plt.title(f'Actual vs Predicted {pollutant}')
    plt.show()

if __name__ == "__main__":
    for pollutant in ("pm10", "pm2.5", "co2", "nox"):
        gbm_pollution(pollutant, remove_outliers=True)