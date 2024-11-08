import matplotlib.pyplot as plt


def plot_test_predictions(y_test, y_pred, save_path, title, figsize):
    plt.figure(figsize=figsize)
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
