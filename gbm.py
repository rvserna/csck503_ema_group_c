import pandas as pd
import matplotlib.pyplot as plt

# load the final dataset
data_path = r"C:\my-project\csck503_ema\csck503_ema_group_c\parsed_data\final_df.pkl"
df = pd.read_pickle(data_path)

# specify which columns will be features and labels
# total pollutant levels will be labels (y), which the model will try to predict
labels = [
    "total_pollutant_co2_2019",
    "total_pollutant_nox_2019",
    "total_pollutant_pm10_2019",
    "total_pollutant_pm2.5_2019"
]
# all other columns will be features (X), which will inform the model's predictions
X = df.drop(columns=labels)

y = df[labels]

# split data into 2 sets, training and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize & train the gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor

gbm_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_regressor.fit(X_train, y_train)

# predict on the test set
y_pred = gbm_regressor.predict(X_test)

# use mean squared error and r-squared to evaluate the model's performance
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# create a visualization of the model's performance
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()