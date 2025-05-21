# Rossmann Time Series Analysis

## Project Overview

This project aims to predict daily sales for Rossmann stores across Germany using time series analysis techniques. Accurate sales forecasts enable store managers to optimize staff scheduling, inventory management, and enhance customer satisfaction.

We implement an **ARIMA** ([AutoRegressive Integrated Moving Average](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)) model to perform the forecasting and compare its performance with a Linear Regression model and a Dummy model that predicts the mean sales.

## Dataset Description

The dataset is sourced from the [Rossmann Store Sales competition on Kaggle](https://www.kaggle.com/c/rossmann-store-sales). It contains historical sales data for over 1,000 Rossmann stores, including information such as sales, customers, promotions, and store details.

## Time Series Forecasting with ARIMA

The **[ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)** (AutoRegressive Integrated Moving Average) model is a powerful statistical method for time series forecasting. It combines three components:

- **[AutoRegression (AR)](https://www.statisticshowto.com/autoregressive-model/)**: Uses the relationship between an observation and a number of lagged observations.
- **[Integrated (I)](https://www.statisticshowto.com/integrated-time-series/)**: Involves differencing of observations to make the time series stationary.
- **[Moving Average (MA)](https://www.statisticshowto.com/moving-average/)**: Uses dependency between an observation and residual errors from moving average models applied to lagged observations.

For more information on ARIMA models, refer to this [comprehensive guide](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/).

### Key Concepts:

- **[Stationarity](https://www.statisticshowto.com/stationary-process/)**: A time series is stationary if its statistical properties like mean and variance are constant over time.
- **[Differencing](https://otexts.com/fpp2/differencing.html)**: A method of transforming a time series dataset by subtracting the previous observation from the current observation.
- **[Seasonality](https://en.wikipedia.org/wiki/Seasonality)**: Patterns that repeat at regular intervals due to seasonal factors.


## Project Structure

```
rossmann-time-series-analysis/
│
├── config/
│   └── config.yaml
│
├── models/
│   ├── base_model.py
│   ├── arima_model.py
│   ├── linear_regression_model.py
│   └── dummy_model.py
│
├── src/
│   ├── data.py
│   ├── plots.py
│   ├── setup_logging.py
│   └── constants.py
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_plots.py
│
├── results/
│   └── arima_test_predictions.png
│
├── main.py
├── requirements.txt
└── README.md
```

## Installation

To run this project, ensure you have Python 3.7 or higher installed. Follow these steps to set up the environment:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/rossmann-time-series-analysis.git
   cd rossmann-time-series-analysis
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Download the dataset** from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and place it in the `data/` directory.

6. **Run the main script**:

   ```bash
   python main.py
   ```

## Configuration

The project uses a `config/config.yaml` file to store configurable parameters, allowing easy adjustments without modifying the codebase directly. Key configurations include:

- **Data Paths**: Specify the path to the dataset (`data_path`) and where to save results (`results_dir_name`).
- **Model Parameters**:
  - **ARIMA Parameters** (`arima_params`): Settings for the ARIMA model, such as seasonal periods (`m`), and maximum orders (`max_p`, `max_d`, `max_q`).
  - **Linear Regression Parameters** (`linear_regression_params`): Options like `fit_intercept` and `normalize`.
  - **Dummy Model Parameters** (`dummy_model_params`): Strategy for the Dummy model, e.g., predicting the mean or median.
- **Plotting Options** (`plot_params`): Customize plot titles (`title`) and figure sizes (`figsize`).

## Data Preprocessing

Data preprocessing involves cleaning and transforming raw data to make it suitable for modeling. The steps include:

- **Data Loading**: Reading the dataset from CSV files.
- **Data Cleaning**:
  - Handling missing values.
  - Encoding categorical variables.
- **Feature Engineering**:
  - Aggregating data at the required granularity.
  - Creating new features like moving averages or lag features.
- **Train-Test Split**: Splitting the data into training and testing sets based on a specified percentage.

The preprocessing is implemented in `src/data.py`.

## Models Implemented

### BaseModel

An abstract base class `BaseModel` is defined in `models/base_model.py`, providing a consistent interface for all models:

- `fit(X_train, y_train)`
- `predict(X_test)`
- `evaluate(X_test, y_test)`

All models inherit from `BaseModel`, ensuring consistent method signatures and enabling polymorphism.

### ARIMAModel

Located in `models/arima_model.py`, this model implements the ARIMA forecasting method using the `statsmodels` library. It automatically determines the optimal parameters using the `pmdarima` library's `auto_arima` function.

### LinearRegressionModel

Implemented in `models/linear_regression_model.py`, this model uses scikit-learn's `LinearRegression` to perform regression on the time series data. While not specifically designed for time series data, it serves as a baseline for comparison.

### DummyModel

Found in `models/dummy_model.py`, this model uses scikit-learn's `DummyRegressor` to make predictions using simple strategies like predicting the mean value of the training data.

## Plotting Function

The `plot_test_predictions` function in `src/plots.py` is used to visualize the actual vs. predicted values for the test set. It uses Matplotlib to generate the plots.

Parameters:

- `y_test`: Actual values.
- `y_pred`: Predicted values.
- `save_path`: File path to save the plot.
- `title`: Plot title (configured in `config.yaml`).
- `figsize`: Figure size (configured in `config.yaml`).

## Logging and Testing

The project includes logging to track the execution flow and debug issues easily. Logs are saved to a file and output to the console, configured via `src/setup_logging.py`.

Unit tests are provided in the `tests/` directory to ensure the reliability and correctness of the data preprocessing, models, and plotting functions.

## Results

After training and evaluating the models, the following RMSE (Root Mean Square Error) values were obtained:

| Model               | RMSE |
|---------------------|------|
| **ARIMA**           | 440.23  |
| Linear Regression   | 626.99  |
| Dummy Model (Mean)  | 813.10  |

The ARIMA model achieved an RMSE of **440.23**, indicating its effectiveness in capturing the underlying patterns in the time series data.

### ARIMA Model Predictions

![ARIMA Model Predictions vs Actual](results/arima_test_predictions.png)

*An example plot showing the ARIMA model's predictions versus the actual sales.*

## Conclusion

The ARIMA model outperformed the Linear Regression and Dummy models in forecasting daily sales for Rossmann stores. This demonstrates the strength of time series-specific models like ARIMA in capturing temporal dependencies and seasonality in data.

**Key takeaways:**

- **ARIMA's Strength**: ARIMA effectively models the time series data, accounting for trends and seasonality.
- **Baseline Comparisons**: The Dummy model and Linear Regression model provide baselines to assess the performance of the ARIMA model.
- **Model Selection**: Linear Regression may not be optimal for time series data without incorporating time-dependent features.

**Future Improvements:**

- **Include Exogenous Variables**: Incorporating additional variables like promotions, holidays, and competitor information could further improve model performance.
- **Advanced Models**: Experimenting with models like SARIMAX, Prophet, or LSTM neural networks.
- **Hyperparameter Tuning**: Further tuning of model parameters using grid search or Bayesian optimization.
- **Cross-Validation**: Implementing time series cross-validation techniques to better assess model performance.

## References

- [Rossmann Store Sales Competition - Kaggle](https://www.kaggle.com/c/rossmann-store-sales)
- [ARIMA Model - Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
- [Time Series Forecasting with ARIMA - Machine Learning Mastery](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
- [Stationarity in Time Series](https://www.statisticshowto.com/stationary-process/)
- [Differencing in Time Series](https://otexts.com/fpp2/differencing.html)
- [Seasonality - Wikipedia](https://en.wikipedia.org/wiki/Seasonality)
- [Scikit-learn DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- [Scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
