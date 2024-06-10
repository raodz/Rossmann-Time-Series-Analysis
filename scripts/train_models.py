import os
from src.data import preprocess_data
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from pmdarima import auto_arima
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError

# Preparing data
X_train, y_train, X_test, y_test = preprocess_data('train.csv')
y_test_mean = y_test.mean()
y_test_sd = y_test.std()
print(f'Sales mean (test): {y_test_mean}')
print(f'Sales standard deviation (test): {y_test_sd}')

# Models
results = {}

# ARIMA
NUM_WEEKS_IN_DATA = 135

seasonality_period = NUM_WEEKS_IN_DATA
autoarima_params = auto_arima(y=y_train.values.reshape(-1),
                              X=X_train,
                              m=seasonality_period,
                              out_of_sample_size=len(X_test),
                              suppress_warnings=True)

autoarima = sm.tsa.arima.model.ARIMA(X_train.values.reshape(-1), order=autoarima_params.order,
                                     seasonal_order=autoarima_params.seasonal_order).fit()

y_pred_autoarima = autoarima.forecast(steps=len(X_test))
auar_rmse = np.sqrt(mean_squared_error(y_test, y_pred_autoarima))
results['Auto ARIMA'] = auar_rmse
print(f'Auto ARIMA RMSE: {auar_rmse}')

# AdaBoost
adaboost = AdaBoostRegressor(n_estimators=100, random_state=0)
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
adab_rmse = np.sqrt(mean_squared_error(y_test, y_pred_adaboost))
results['AdaBoostRegressor'] = adab_rmse
print(f'AdaBoostRegressor RMSE: {adab_rmse}')

# Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
results['Linear Regression'] = reg_rmse
print(f'Linear Regression RMSE: {reg_rmse}')

# Lasso Regression
lasso = Lasso(alpha=.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
results['Lasso'] = lasso_rmse
print(f'Lasso RMSE: {lasso_rmse}')

# Ridge Regression
ridge = Ridge(alpha=.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
results['Ridge'] = ridge_rmse
print(f'Ridge RMSE: {ridge_rmse}')

# Sequential Neural Network Model
model = tf.keras.Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[RootMeanSquaredError()])

model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

loss, rmse = model.evaluate(X_test, y_test)
results['Sequential Model'] = rmse
print(f"Sequential Model RMSE: {rmse}")

# Generate markdown document
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results')
os.makedirs(results_dir, exist_ok=True)
md_path = os.path.join(results_dir, 'model_results.md')

# Find model with lowest RMSE
best_model = min(results, key=results.get)

with open(md_path, 'w') as md_file:
    md_file.write('# Model Results\n\n')
    md_file.write(f'**Mean of y_test:** {y_test_mean:.4f}\n\n')
    md_file.write(f'**Standard Deviation of y_test:** {y_test_sd:.4f}\n\n')
    md_file.write('| Model | RMSE |\n')
    md_file.write('|-------|------|\n')
    for model, rmse in results.items():
        md_file.write(f'| {model} | {rmse:.4f} |\n')
    md_file.write('\n')
    md_file.write(f'**Model with lowest RMSE:** {best_model} ({results[best_model]:.4f})\n')

print(f'Results saved to {md_path}')
