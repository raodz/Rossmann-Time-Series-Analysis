from preprocess_data import X_train, y_train, X_test, y_test
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import statsmodels as sm

adaboost = AdaBoostRegressor(n_estimators=100, random_state=0)
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
adab_rmse = np.sqrt(mean_squared_error(y_test, y_pred_adaboost))
print(f'AdaBoostRegressor RMSE: {adab_rmse}')

autoarima = auto_arima(X_train.values.reshape(-1),
                       out_of_sample_size=len(X_test),
                       suppress_warnings=True)

autoarima = sm.tsa.arima.model.ARIMA(X_train.values.reshape(-1), order=autoarima.order,
                                     seasonal_order=autoarima.seasonal_order).fit()
y_pred_autoarima = autoarima.get_forecast(steps=len(X_test))
auar_rmse = np.sqrt(mean_squared_error(y_test, y_pred_autoarima))
print(f'Auto ARIMA RMSE: {auar_rmse}')

"""
Po linijce 21 taki błąd:
 File "C:\Users\raodz\PycharmProjects\rossmann\.venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
    raise InvalidParameterError(
sklearn.utils._param_validation.InvalidParameterError: The 'y_pred' parameter of mean_squared_error must be an array-like. Got <statsmodels.tsa.statespace.mlemodel.PredictionResultsWrapper object at 0x0000025A567A29C0> instead.

"""