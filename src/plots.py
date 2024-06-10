import matplotlib.pyplot as plt
import pandas as pd


def plot_train_test_predictions(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                                y_test: pd.Series, y_pred: pd.Series, save_path: str = None) -> None:
    """
    Plot actual and predicted values for training and testing data.

    Parameters:
        X_train (pd.DataFrame): DataFrame containing training dates.
        X_test (pd.DataFrame): DataFrame containing testing dates.
        y_train (pd.Series): Series containing actual training values.
        y_test (pd.Series): Series containing actual testing values.
        y_pred (pd.Series): Series containing predicted values.
        save_path (str): Path to directory where plot is saved.

    Returns:
        None
    """
    # Obsługa wyjątku, gdy X_train lub X_test jest pustą ramką danych
    if X_train.empty or X_test.empty:
        raise ValueError("X_train or X_test DataFrame is empty")

    all_dates = pd.concat([pd.Series(X_train.index), pd.Series(X_test.index)])

    plt.figure(figsize=(10, 6))
    plt.plot(X_train.index, y_train, label='Actual (Train)', color='blue')
    plt.plot(X_test.index, y_test, label='Actual (Test)', color='green')
    plt.plot(X_test.index, y_pred, label='Predicted', color='red')

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Value')

    x_ticks = all_dates[::50]
    plt.xticks(x_ticks, rotation=45)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_test_predictions(X_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, save_path: str = None) -> None:
    """
    Plot actual and predicted values for testing data.

    Parameters:
        X_test (pd.DataFrame): DataFrame containing testing dates.
        y_test (pd.Series): Series containing actual testing values.
        y_pred (pd.Series): Series containing predicted values.
        save_path (str): Path to directory where plot is saved.

    Returns:
        None
    """
    if X_test.empty:
        raise ValueError("X_test DataFrame is empty")

    plt.figure(figsize=(10, 6))
    plt.plot(X_test.index, y_test, label='Actual', color='green')
    plt.plot(X_test.index, y_pred, label='Predicted', color='red')

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path)
    plt.show()
