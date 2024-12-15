import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from ml_utils import remove_outliers_by_iqr


def linear_regression_pollution(pollutant, include_other_pollutants=False, remove_outliers=False):
    df = pd.read_pickle("./parsed_data/final_df.pkl")

    # Drop any irrelevant features
    df.drop(columns="grid_id", inplace=True)

    # Select relevant variables
    dependent_variable = f"total_pollutant_{pollutant}_2019"
    # Select features (anything that's not a pollutant)
    if include_other_pollutants:
        independent_variables = [col for col in df.columns if col != dependent_variable]
    else:
        independent_variables = [col for col in df.columns if not col.startswith("total_pollutant")]

    # Remove outliers if appropriate
    if remove_outliers:
        df = remove_outliers_by_iqr(df, [dependent_variable])

    X = df[independent_variables]
    y = df[[dependent_variable]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    regr = LinearRegression()
    regr.fit(X_train_scaled, y_train_scaled)

    # Predict on the scaled test set
    y_pred_scaled = regr.predict(X_test_scaled)

    # Inverse transform the predictions and the true values to the original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    # Calculate performance metrics
    mse = mean_squared_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    # Print performance metrics
    print(f"Mean Squared Error for {pollutant}: {mse:.2f}")
    print(f"R-squared for {pollutant}: {r2:.2f}")

    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': regr.coef_})
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
        linear_regression_pollution(pollutant, remove_outliers=True)

