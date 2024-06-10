from src.data import preprocess_data, sliding_window
from sklearn.linear_model import Lasso
from src.plots import plot_test_predictions, plot_train_test_predictions
import numpy as np


def main():
    import pandas as pd
    df = sliding_window(pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10], 'target': [11, 12, 13, 14, 15]}),
                        'target', 3)
    print(type(df[0]['A'][1]))
    # Data
    # state_holiday_mapping = {'0': 0, 'a': 1, 'b': 1, 'c': 1}  # All types of holidays are mapped into 1
    # X_train, y_train, X_test, y_test = preprocess_data('train.csv', state_holiday_mapping)
    #
    # # Model
    # model = Lasso(alpha=.1)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    #
    # zero_coef_indices = np.where(model.coef_ == 0)[0]
    # zero_coef_column_names = [X_test.columns[i] for i in zero_coef_indices]
    # print(f'Zeroed variables: {zero_coef_column_names}')
    #
    # # Plots
    #
    # plot_test_predictions(X_test, y_test, y_pred)
    # plot_train_test_predictions(X_train, X_test, y_train, y_test, y_pred)


if __name__ == "__main__":
    main()
