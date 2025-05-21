import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_test_predictions(y_test, y_pred, save_path, title, figsize):
    """Plot actual vs predicted values for time series data and save to file."""
    plt.figure(figsize=figsize)

    if not isinstance(y_test.index, pd.DatetimeIndex):
        y_test = y_test.copy()
        y_test.index = pd.to_datetime(y_test.index)

    plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2, color='#3498db')
    plt.plot(y_test.index, y_pred, label='Predicted', linewidth=2, color='#e74c3c', linestyle='--')

    ax = plt.gca()
    n = max(1, len(y_test) // 10)  # Show approximately 10 dates on the axis

    date_fmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_fmt)

    if len(y_test) > 30:
        plt.xticks(y_test.index[::n])  # every nth date
    else:
        plt.xticks(y_test.index)

    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()