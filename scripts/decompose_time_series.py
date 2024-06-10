from src.data import preprocess_data
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import os

# Prepare data
X_train, y_train, X_test, y_test = preprocess_data('train.csv')

# Decompose time series
result = seasonal_decompose(y_train, model='additive', period=30)

# Create results directory if it doesn't exist
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results')
os.makedirs(results_dir, exist_ok=True)

# Plot and save the decomposition results
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(y_train, label='Original')
plt.legend(loc='upper left')
plt.xticks([])  # Remove xticks for this subplot

plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend(loc='upper left')
plt.xticks([])  # Remove xticks for this subplot

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.xticks([])  # Remove xticks for this subplot

plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residuals')
plt.legend(loc='upper left')
plt.xticks(ticks=range(0, len(y_train), 50), labels=[y_train.index[i] for i in range(0, len(y_train), 50)], rotation=45)

plt.suptitle('Time Series Decomposition')

# Save the plot
plot_path = os.path.join(results_dir, 'decomposition_plot.png')
plt.savefig(plot_path)
plt.show()

print(f'Plot saved to {plot_path}')
