from src.data import preprocess_data
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from pmdarima import auto_arima
import statsmodels as sm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError


# Preparing data
X_train, y_train, X_test, y_test = preprocess_data('train.csv')
print(f'Sales mean: {y_train.mean()}')
print(X_train)

# Models

# ARIMA

# NUM_WEEKS_IN_DATA = 135
#
# seasonality_period = NUM_WEEKS_IN_DATA
# autoarima_params = auto_arima(y=y_train.values.reshape(-1),
#                               X=X_train,
#                               m=seasonality_period,
#                               out_of_sample_size=len(X_test),
#                               suppress_warnings=True)
#
# autoarima = sm.tsa.arima.model.ARIMA(X_train.values.reshape(-1), order=autoarima_params.order,
#                                      seasonal_order=autoarima_params.seasonal_order).fit()
#
# y_pred_autoarima = autoarima.forecast(steps=len(X_test))
# auar_rmse = np.sqrt(mean_squared_error(y_test, y_pred_autoarima))
# print(f'Auto ARIMA RMSE: {auar_rmse}')

# AdaBoost

adaboost = AdaBoostRegressor(n_estimators=100, random_state=0)
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
adab_rmse = np.sqrt(mean_squared_error(y_test, y_pred_adaboost))
print(f'AdaBoostRegressor RMSE: {adab_rmse}')

# Linear Regression

reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred_reg = reg.predict(X_test)
reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
print(f'Linear Regression RMSE: {reg_rmse}')

# Lasso Regression

lasso = Lasso(alpha=.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f'Lasso RMSE: {lasso_rmse}')

# Ridge Regression

ridge = Ridge(alpha=.1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
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
print(f"Sequential Model RMSE: {rmse}")
