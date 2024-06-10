from src.data import preprocess_data
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

state_holiday_mapping = {'0': 0, 'a': 1, 'b': 1, 'c': 1}  # All types of holidays are mapped into 1
X_train, y_train, X_test, y_test = preprocess_data('train.csv', state_holiday_mapping)

result = seasonal_decompose(y_train, model='additive', period=30)

plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(y_train, label='Original')
plt.legend(loc='upper left')

plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonality')
plt.legend(loc='upper left')

plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residuals')
plt.legend(loc='upper left')

plt.suptitle('Dekompozycja Szeregu Czasowego')
plt.show()
