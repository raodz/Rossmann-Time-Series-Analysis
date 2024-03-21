from src.preprocessing import preprocess_data
from train_models import state_holiday_mapping
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest

X_train, y_train, X_test, y_test = preprocess_data('train.csv', state_holiday_mapping)

result = seasonal_decompose(y_train, model='additive', period=30)

# Wykresy składowych

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

# Usuwanie trendu w celu otrzymania szeregu stacjonarnego

cycle, trend = sm.tsa.filters.hpfilter(y_train, lamb=129600)

stationary_catfish = cycle

plt.figure(figsize=(12, 2))
plt.plot(stationary_catfish, label='Dane w wersji stacjonarnej')
plt.legend(loc='upper left')
plt.show()

# Isolation Forest

model = IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.02),random_state=241)

model.fit(stationary_catfish.to_numpy().reshape(-1, 1))

# Wykrywanie anomalii za pomocą metody predict i zaznaczenie ich na wykresie

anomalies = model.predict(stationary_catfish.to_numpy().reshape(-1, 1))

anomalies_idx = [i for i, x in enumerate(anomalies) if x == -1]

plt.figure(figsize=(12, 2))
plt.plot(X_train, c ="b")
plt.scatter(y_train[anomalies_idx].index, y_train[anomalies_idx], c="r")
plt.legend(loc='upper left')
plt.show()
