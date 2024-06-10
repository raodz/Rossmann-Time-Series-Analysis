import os
from src.data import preprocess_data
from src.plots import plot_test_predictions, plot_train_test_predictions
from sklearn.linear_model import Lasso
import numpy as np
import joblib


def main():

    # Create results directory if it doesn't exist
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Data
    X_train, y_train, X_test, y_test = preprocess_data('train.csv')

    # Model
    model = Lasso(alpha=.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    zero_coef_indices = np.where(model.coef_ == 0)[0]
    zero_coef_column_names = [X_test.columns[i] for i in zero_coef_indices]
    print(f'Zeroed variables: {zero_coef_column_names}')

    # Save model
    model_path = os.path.join(results_dir, 'lasso_model.joblib')
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

    # Plots
    plot_test_path = os.path.join(results_dir, 'test_predictions.png')
    plot_test_predictions(X_test, y_test, y_pred, save_path=plot_test_path)
    print(f'Test predictions plot saved to {plot_test_path}')

    plot_train_test_path = os.path.join(results_dir, 'train_test_predictions.png')
    plot_train_test_predictions(X_train, X_test, y_train, y_test, y_pred, save_path=plot_train_test_path)
    print(f'Train and test predictions plot saved to {plot_train_test_path}')


if __name__ == "__main__":
    main()
