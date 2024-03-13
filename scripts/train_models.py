from src.preprocessing import preprocess_data
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import statsmodels as sm
from src.plots import visualize_arima_results

state_holiday_mapping = {'0': 0, 'a': 1, 'b': 1, 'c': 1}
# All types of holidays are mapped into 1

X_train, y_train, X_test, y_test = preprocess_data('train.csv', state_holiday_mapping)

adaboost = AdaBoostRegressor(n_estimators=100, random_state=0)
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
adab_rmse = np.sqrt(mean_squared_error(y_test, y_pred_adaboost))
print(f'AdaBoostRegressor RMSE: {adab_rmse}')

# seasonality_period = 365

# autoarima_params = auto_arima(y=y_train.values.reshape(-1),
#                               X=X_train,
#                               m=seasonality_period,
#                               out_of_sample_size=len(X_test),
#                               suppress_warnings=True)

# autoarima = sm.tsa.arima.model.ARIMA(X_train.values.reshape(-1), order=autoarima_params.order,
#                                      seasonal_order=autoarima_params.seasonal_order).fit()
# y_pred_autoarima = autoarima.forecast(steps=len(X_test))
# auar_rmse = np.sqrt(mean_squared_error(y_test, y_pred_autoarima))
# print(f'Auto ARIMA RMSE: {auar_rmse}')

# predictions = autoarima.get_forecast(steps=len(X_test))
# visualize_arima_results(y_test, predictions, 10000)